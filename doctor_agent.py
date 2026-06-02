# 文件名: doctor_agent.py

import re
import os
import json
from typing import List, Dict, Callable
from openai import OpenAI
from topic_predictor import TopicPredictor # 导入话题预测模块

# --- SFT 输出解析工具 (保持不变) ---
def _parse_output(raw_text: str) -> Dict:
    """
    解析 <think>...</think><response>...</response> 格式的 SFT 输出。
    """
    # 提取 think
    think_match = re.search(r'<think>(.*?)</think>', raw_text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""
    
    # 从 think 中提取 topic (寻找 topic: 或 topic：)
    topic_match = re.search(r'topic[:：]\s*(.*)', think_content)
    if topic_match:
        topic = topic_match.group(1).strip().split('\n')[0] 
    else:
        topic = "其他" 
        
    # 提取 response
    response_match = re.search(r'<response>(.*?)</response>', raw_text, re.DOTALL)
    response_content = response_match.group(1).strip() if response_match else raw_text
    
    return {
        "think": think_content,
        "topic": topic,
        "response": response_content
    }

# --- Doctor Agent 核心类 ---

class DoctorAgent:
    """
    整合话题预测和 LLM API 调用的 Doctor Agent。
    使用 OpenAI 兼容的 API 接口进行模型生成，并遵循指定 Prompt 格式。
    """
    SYSTEM_INSTRUCTION = (
        "你是一个精神科医生，具有丰富的临床经验。你的目的是通过临床问答来获取患者的信息，"
        "逐步引导患者来获取其与心理相关的特征以便于进行临床诊断，"
        "你需要在think标签中推断接下来的话题，然后在response标签中输出回复。"
    )
    
    def __init__(self, 
                 topic_predictor: TopicPredictor, 
                 model_name: str,
                 api_key: str = "0", 
                 base_url: str = "http://0.0.0.0:8000/v1"):
        
        self.predictor = topic_predictor
        self.model_name = model_name
        
        # 初始化 OpenAI 兼容客户端
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        self.reset_state() 

    def reset_state(self):
        """重置 Agent 的内部状态，用于开始新对话。"""
        self.topic_history: List[str] = []     # 实际执行的话题历史
        self.current_topic: str = "开始"      # 上一轮（或初始）的话题
        # 使用 Llama/Qwen 的 messages 格式来存储历史：[{"role": "user/assistant", "content": "..."}]
        self.dialogue_history: List[Dict] = [] 

    def _build_api_messages(self, predicted_topic: str, user_input: str) -> List[Dict]:
        """
        根据预测话题、系统指令和历史，构建 API 所需的 messages 列表。
        """
        # 1. 话题指令 (作为系统消息的一部分或用户消息的前缀)
        topic_instruction = f"接下来考虑进行的话题为「{predicted_topic}」"
        #print(f"------预测话题--------\n预测的接下来的话题为{topic_instruction}")
        
        # 2. 系统消息
        system_message = {
            "role": "system", 
            "content": f"{self.SYSTEM_INSTRUCTION}"
        }
        
        # 3. 历史消息 (深拷贝防止修改)
        messages = [system_message] + list(self.dialogue_history)
        #print(f"------历史消息--------\n当前的历史话题的集合{messages}")
        
        # 4. 当前用户输入
        messages.append({"role": "user", "content":  f"{topic_instruction}\n{user_input}"})
        #print(f"------整体输入--------\n当前输入doctor的内容{messages}")
        
        return messages

    def _api_call(self, messages: List[Dict]) -> str:
        """
        调用 OpenAI 兼容 API 并返回模型原始回复文本。
        """
        try:
            # 调用 API
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                # 确保 max_tokens 足够长，以包含 think 和 response 标签
                max_tokens=2048, 
                temperature=0.7 # 确保回复具有一定多样性
            )
            raw_response = result.choices[0].message.content
            print(f"------整体输出--------\n医生回复的内容{raw_response}")
            return raw_response
        except Exception as e:
            # 记录 API 错误，返回一个错误标记
            print(f"API 调用失败: {e}")
            return f"<think>API调用失败。topic:其他</think><response>抱歉，我暂时无法回答。错误信息: {e}</response>"

    def step(self, user_input: str) -> Dict:
        """
        执行一轮对话，返回结果和状态更新。
        """
        # 1. 话题预测 (Planning)
        predicted_topic = self.predictor.predict(
            history_topics=self.topic_history, 
            current_topic=self.current_topic
        )
        
        # 2. 构建 API Messages
        messages_to_send = self._build_api_messages(predicted_topic, user_input)
        
        # 3. 生成回复 (Acting)
        raw_output = self._api_call(messages_to_send)
        
        # 4. 解析输出
        parsed_result = _parse_output(raw_output)
        doctor_response = parsed_result['response']
        actual_topic = parsed_result['topic']
        
        # 5. 更新状态 (遵循 Qwen 历史更新逻辑)
        
        # 将用户输入和医生回复添加到历史中
        self.dialogue_history.append({"role": "user", "content": user_input})
        self.dialogue_history.append({"role": "assistant", "content": doctor_response})
        
        # 更新话题历史
        self.topic_history.append(self.current_topic) 
        self.current_topic = actual_topic # SFT 实际执行的话题
        
        return {
            "response": doctor_response,
            "predicted_topic": predicted_topic,
            "actual_topic": actual_topic,
            "think_content": parsed_result['think']
        }

if __name__ == '__main__':
    # =============================================================
    # 单元测试示例 (Mocking TopicPredictor 和 API Call)
    # =============================================================
    
    print("--- 运行 DoctorAgent 单元测试 ---")

    # 1. Mock TopicPredictor (模拟预测器的行为)
    class MockTopicPredictor:
        def __init__(self):
            self.topics = iter(["不上学/间断上学", "成绩", "结束语"])
        def predict(self, history_topics, current_topic):
            # 模拟话题链
            return next(self.topics)

    # 2. Mock API Call (模拟 LLM 的输出)
    def mock_api_call(self, messages):
        # 简单的模拟 LLM 对话逻辑
        user_input = messages[-1]['content']
        predicted_topic = messages[0]['content'].split("为「")[1].split("」")[0]
        
        if predicted_topic == "不上学/间断上学":
            think = f"根据用户输入，我们聚焦于 {predicted_topic}。topic:不上学/间断上学"
            response = "您好，我明白了。能详细说说是从什么时候开始不想上学的吗？"
        elif predicted_topic == "成绩":
            think = f"已收集上学信息，转入学习压力 {predicted_topic}。topic:成绩"
            response = "好的。那么您在学校里的成绩怎么样？有压力吗？"
        else:
            think = f"信息收集完毕。topic:结束语"
            response = "感谢您的配合，主要信息已收集完毕。"

        return f"<think>{think}</think><response>{response}</response>"
        
    # 临时覆盖 _api_call 方法以进行 Mock 测试
    DoctorAgent._api_call = mock_api_call 

    # 3. 实例化 Agent
    mock_predictor = MockTopicPredictor()
    agent = DoctorAgent(
        topic_predictor=mock_predictor, 
        model_name="Qwen3-8B", 
        base_url="http://mock-url"
    )

    # --- 对话测试 ---
    
    # Turn 1:
    user_input_1 = "医生，我最近不想上学，心情很差。"
    result_1 = agent.step(user_input_1)
    print(f"\n[T1 Input]: {user_input_1}")
    print(f"[T1 Result]: Pred={result_1['predicted_topic']}, Actual={result_1['actual_topic']}")
    print(f"[T1 Response]: {result_1['response']}")
    assert result_1['actual_topic'] == "不上学/间断上学"
    
    # Turn 2:
    user_input_2 = "从初一开始的，没什么特别的。"
    result_2 = agent.step(user_input_2)
    print(f"\n[T2 Input]: {user_input_2}")
    print(f"[T2 Result]: Pred={result_2['predicted_topic']}, Actual={result_2['actual_topic']}")
    print(f"[T2 Response]: {result_2['response']}")
    assert result_2['actual_topic'] == "成绩"
    
    # Turn 3: 检查历史话题
    print(f"\n[T3 Check]: 话题历史: {agent.topic_history}")
    assert agent.topic_history == ["开始", "不上学/间断上学"] 
    
    # Turn 3: 结束语
    user_input_3 = "一般吧，中等。"
    result_3 = agent.step(user_input_3)
    print(f"\n[T3 Input]: {user_input_3}")
    print(f"[T3 Result]: Pred={result_3['predicted_topic']}, Actual={result_3['actual_topic']}")
    print(f"[T3 Response]: {result_3['response']}")
    assert result_3['actual_topic'] == "结束语"
    
    print("\nDoctorAgent 单元测试通过 (基于 Mocking)")
# 文件名: doctor_agent.py (底部)

if __name__ == '__main__':
    # =============================================================
    # 💥 实时 API 连接测试 (DoctorAgent 单元测试) 💥
    # 运行此代码前，请确保您的 Llama/OpenAI 兼容服务正在运行。
    # =============================================================
    
    print("--- 运行 DoctorAgent 实时 API 连接测试 ---")

    # 1. 配置您的 API 详情 (!!! 必须修改 !!!)
    TEST_MODEL_NAME = "Qwen3-8B" # 替换为您的模型名称
    # 替换为您的实际 API 地址
    TEST_BASE_URL = "http://0.0.0.0:8000/v1" 
    TEST_API_KEY = "0" 
    
    # 2. Mock TopicPredictor (只返回一个固定话题，避免加载 BERT)
    class MockTopicPredictor:
        def __init__(self, fixed_topic="不上学/间断上学"):
            self.fixed_topic = fixed_topic
            print(f"[Mock] 话题预测器已启动，将始终预测话题: {self.fixed_topic}")
        def predict(self, history_topics, current_topic):
            return self.fixed_topic

    # 3. 实例化 Agent
    mock_predictor = MockTopicPredictor()
    agent = DoctorAgent(
        topic_predictor=mock_predictor, 
        model_name=TEST_MODEL_NAME, 
        api_key=TEST_API_KEY,
        base_url=TEST_BASE_URL
    )

    # --- 对话测试 (仅一轮) ---
    try:
        user_input_1 = "医生，我最近心情很差，不想去学校。"
        print(f"\n[测试输入]: {user_input_1}")
        
        # 执行 step 方法，会触发实际的 API 调用
        result_1 = agent.step(user_input_1)
        
        print("\n--- 结果验证 ---")
        print(f"[API 状态]: 连接成功")
        print(f"[预测话题]: {result_1['predicted_topic']}")
        print(f"[实际话题]: {result_1['actual_topic']}")
        print(f"[Think 内容]: {result_1['think_content']}")
        print(f"[医生回复]:\n{result_1['response']}")
        print("------------------")

        # 验证历史是否更新
        assert len(agent.dialogue_history) == 2 
        print(f"[历史轮次]: {len(agent.dialogue_history) // 2} 轮对话历史已记录。")

    except Exception as e:
        print(f"\n--- 错误报告 ---")
        print(f"致命错误：Agent 测试失败。请检查 API 服务是否在 {TEST_BASE_URL} 上运行。")
        print(f"错误详情: {e}")
        print("------------------")F
