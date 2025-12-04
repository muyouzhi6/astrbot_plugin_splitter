import re
import math
import random
import asyncio
from typing import List

from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger
from astrbot.api.provider import LLMResponse
from astrbot.api.message_components import Plain, BaseMessageComponent

@register("astrbot_plugin_splitter", "YourName", "LLM 输出智能分段发送插件", "1.1.0")
class MessageSplitterPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # 定义成对符号，Key为左符号，Value为右符号
        self.pair_map = {
            '“': '”', 
            '《': '》', 
            '（': '）', 
            '(': ')', 
            '[': ']', 
            '{': '}',
            '"': '"', # 英文引号特殊处理
            "'": "'"  # 英文单引号特殊处理
        }

    # 1. 标记阶段
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        setattr(event, "__is_llm_reply", True)

    # 2. 拦截与处理阶段
    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        # 检查是否为 LLM 回复
        if not getattr(event, "__is_llm_reply", False):
            return

        # 检查是否已经处理过（防止重复日志/发送）
        if getattr(event, "__splitter_processed", False):
            return
        setattr(event, "__splitter_processed", True) # 加锁

        result = event.get_result()
        if not result or not result.chain:
            return

        # 获取配置
        split_pattern = self.config.get("split_regex", r"[。？！?!\\n…]+")
        smart_mode = self.config.get("enable_smart_split", True)

        # 执行分段
        segments = self.split_chain_smart(result.chain, split_pattern, smart_mode)

        # 如果只有一段，直接放行，不干预
        if len(segments) <= 1:
            return

        logger.info(f"[Splitter] 检测到长文本，已智能分割为 {len(segments)} 段。")

        # 逐段发送
        for i, segment_chain in enumerate(segments):
            if not segment_chain:
                continue

            # 提取纯文本用于日志和延迟计算
            text_content = "".join([c.text for c in segment_chain if isinstance(c, Plain)])
            logger.info(f"[Splitter] 发送第 {i+1} 段 (len={len(text_content)}): 已分段文本：{text_content}")

            try:
                mc = MessageChain()
                mc.chain = segment_chain
                await self.context.send_message(event.unified_msg_origin, mc)

                # 计算并执行延迟（如果是最后一段则不需要延迟）
                if i < len(segments) - 1:
                    wait_time = self.calculate_delay(text_content)
                    await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"[Splitter] 发送分段失败: {e}")

        # 清空原始消息链，阻止 AstrBot 发送原始大段消息，但保留历史记录流程
        result.chain.clear()

    def calculate_delay(self, text: str) -> float:
        """根据策略计算延迟时间"""
        strategy = self.config.get("delay_strategy", "log")
        
        if strategy == "random":
            mn = self.config.get("random_min", 1.0)
            mx = self.config.get("random_max", 3.0)
            return random.uniform(mn, mx)
            
        elif strategy == "log":
            # 拟人化延迟：Base + Factor * ln(len + 1)
            # 文本越长延迟越久，但增长速率逐渐放缓
            base = self.config.get("log_base", 0.5)
            factor = self.config.get("log_factor", 0.8)
            length = len(text)
            # 防止过长文本导致等待太久，设置一个硬上限（比如5秒）
            delay = base + factor * math.log(length + 1)
            return min(delay, 5.0) 
            
        else: # fixed
            return self.config.get("fixed_delay", 1.5)

    def split_chain_smart(self, chain: List[BaseMessageComponent], pattern: str, smart_mode: bool) -> List[List[BaseMessageComponent]]:
        """
        智能分段核心逻辑：
        1. 遍历组件，非Plain组件直接归入当前段。
        2. Plain组件按字符解析（如果开启Smart Mode），保护成对符号内的内容。
        3. 连续的分隔符会被合并并附着在上一段末尾。
        """
        segments = []
        current_chain_buffer = []

        for component in chain:
            if not isinstance(component, Plain):
                current_chain_buffer.append(component)
                continue

            text = component.text
            if not text:
                continue

            if not smart_mode:
                # 兼容旧模式（仅正则，但优化了连续分隔符处理）
                self._process_text_simple(text, pattern, segments, current_chain_buffer)
            else:
                # 智能模式
                self._process_text_smart(text, pattern, segments, current_chain_buffer)

        # 处理末尾剩余内容
        if current_chain_buffer:
            segments.append(current_chain_buffer)

        # 清理空段
        return [seg for seg in segments if seg]

    def _process_text_simple(self, text: str, pattern: str, segments: list, buffer: list):
        """简单的正则分段逻辑，处理连续分隔符"""
        # 使用捕获组保留分隔符
        parts = re.split(f"({pattern})", text)
        
        temp_text = ""
        for part in parts:
            if not part: continue
            
            # 检查这部分是不是完全由分隔符组成
            if re.fullmatch(pattern, part):
                # 是分隔符，附着在当前 accum_text 后
                temp_text += part
                # 遇到分隔符，结算一次
                buffer.append(Plain(temp_text))
                segments.append(buffer[:]) # Shallow copy
                buffer.clear() # Clear reference
                temp_text = ""
            else:
                # 是普通文本
                if temp_text:
                    # 如果之前有未结算的文本（理论上不应该，除非连续两个文本块），先入buffer
                    buffer.append(Plain(temp_text))
                temp_text = part
        
        if temp_text:
            buffer.append(Plain(temp_text))

    def _process_text_smart(self, text: str, pattern: str, segments: list, buffer: list):
        """
        智能逐字解析
        """
        stack = [] # 用于存储左括号/引号
        compiled_pattern = re.compile(pattern)
        
        i = 0
        n = len(text)
        current_chunk = ""

        while i < n:
            char = text[i]

            # 1. 检查是否是成对符号的处理
            is_opener = char in self.pair_map
            # 对于英文引号 "，既是左也是右，需要特殊判断
            if char in ['"', "'"]:
                if stack and stack[-1] == char:
                    # 栈顶是自己，说明是右引号 -> 出栈
                    stack.pop()
                    current_chunk += char
                    i += 1
                    continue
                else:
                    # 栈顶不是自己，说明是左引号 -> 入栈
                    stack.append(char)
                    current_chunk += char
                    i += 1
                    continue
            
            # 处理普通成对符号
            if stack:
                expected_closer = self.pair_map.get(stack[-1])
                if char == expected_closer:
                    stack.pop() # 匹配闭合，出栈
                elif is_opener:
                    stack.append(char) # 嵌套，入栈
                
                # 无论如何，在栈内时，字符都视为普通内容
                current_chunk += char
                i += 1
                continue
            
            # 2. 如果不在栈内（不在引用中）
            if is_opener:
                stack.append(char)
                current_chunk += char
                i += 1
                continue

            # 3. 检查是否匹配分隔符（lookahead check）
            # 我们需要检查当前位置是否是分隔符的开始
            match = compiled_pattern.match(text, pos=i)
            
            if match:
                # 匹配到了分隔符！
                delimiter = match.group()
                
                # 将前面的文本 + 分隔符 一起加入 buffer
                current_chunk += delimiter
                buffer.append(Plain(current_chunk))
                
                # 生成一个分段
                segments.append(buffer[:]) # Copy
                buffer.clear() # Clear
                
                current_chunk = ""
                i += len(delimiter) # 跳过分隔符长度
            else:
                # 普通字符
                current_chunk += char
                i += 1

        # 循环结束，如果有剩余文本留在 current_chunk
        if current_chunk:
            buffer.append(Plain(current_chunk))
