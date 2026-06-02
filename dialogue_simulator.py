# 文件名: evaluation_main.py

import os
import json
import jsonlines
import time
import torch
from typing import Dict, List, Callable
from openai import OpenAI
import re # 用于 Patient Agent 的 Mock 逻辑

# 导入我们已验证的模块
from topic_predictor import TopicPredictor
from doctor_agent import DoctorAgent
import random

# =============================================================
# 1. 配置和 LLM/SFT 占位函数 (请替换为您的实际实现)
# =============================================================

# 定义 Patient 的风格列表，用于循环选择
PATIENT_STYLES = ["内向", "中性", "外向"]


# =============================================================
# 1. 配置和 LLM/SFT 占位函数 (请替换为您的实际实现)
# =============================================================

# 定义 Patient 的风格列表，用于循环选择
PATIENT_STYLES = ["内向", "中性", "健谈"]


# --- A. Patient LLM 调用函数 (Patient Agent) ---
def actual_patient_llm_call(
    system_prompt: str, 
    doctor_response: str, 
    dialogue_history: List[Dict], # <-- 新增：接收 Doctor Agent 维护的历史记录
    client: OpenAI, 
    model_name: str
) -> str:
    """ 
    使用 OpenAI 兼容 API 调用 Patient LLM (GPT-3.5-turbo/GPT-4o) 生成回复。
    历史记录将被用于 API 调用。
    """
    
    # 1. 构建 API 所需的 messages 列表
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # 2. 加入历史记录 (Patient LLM 视角)
    # Doctor Agent 的 history 格式是 {"role": "user/assistant", "content": "..."}
    # 刚好可以直接用于 GPT API 的 messages
    
    # 注意：dialogue_history 已经是 Doctor Agent 维护的完整历史。
    # 我们应该只传入完整的历史，然后将 Doctor Agent 的最新回复作为 LLM 的上一个 Assistant 回复。
    
    # 假设 Doctor Agent 的 dialogue_history 已经包含【当前轮次之前】的所有对话。
    # 格式应该是：[P1, D1, P2, D2, ..., P(n-1), D(n-1)]
    messages.extend(dialogue_history)
    
    # 3. 将 Doctor Agent 的最新回复（doctor_response）作为当前轮次的【用户输入】发给 Patient LLM
    # 因为 Patient LLM 扮演的是 "Patient"，它需要接收"医生"的回复。
    messages.append({
        "role": "user", 
        "content": f"医生回复：{doctor_response}"
    })


    try:
        # 调用 API
        response = client.chat.completions.create(
            model=model_name, 
            messages=messages, # <-- 关键：传入包含历史的 messages
            temperature=0.7,
            max_tokens=512
        )
        
        result = response.choices[0].message.content
        return result
    
    except Exception as e:
        print(f"Patient LLM API 调用失败 ({model_name}): {e}")
        return "抱歉，我听不清您说什么。"
    
# -------------------------------------------------------------
# 2. 运行配置与辅助函数
# -------------------------------------------------------------
LOG_DIR = "evaluation_logs"
MAX_TURNS = 50 
# 👇👇👇 必须修改 👇👇👇
PATIENT_DATA_FILE = "../data/extracted_dialog_data.jsonl" 
#PATIENT_DATA_FILE = "../data/patient_test.jsonl" 

# Doctor Agent/SFT 模型配置 (与 doctor_agent.py 中使用的 API 需保持一致)
SFT_MODEL_CONFIG = {
    'model_name': "Qwen3-8B",
    'api_key': "0",
    'base_url': "http://localhost:8000/v1"
}

# Topic Predictor 配置
TOPIC_PREDICTOR_CONFIG = {
    # 训练完成的topic预测器
    'bert_path': '../data/model_epoch_10.pt',
    'matrix_path': '../data/transition_martrix.csv',
}

# Patient LLM 配置 (使用您指定的 GPT-3.5-turbo/GPT-4o)
PATIENT_LLM_CONFIG = {
    # 
    'model_name': " ", # 如果使用 GPT-4o，请改为 "gpt-4o"
    # 建议从环境变量获取，或者手动填写您的 API Key
    "api_key" : "api_key",
    "base_url" : "your_base_url" 
}


def ensure_directory(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    print(f"日志目录已就绪: {path}")


def load_patient_data(file_path: str) -> List[Dict]:
    """加载 JSONL 格式的患者数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"患者数据文件不存在: {file_path}。请检查路径。")
        
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data 

# -------------------------------------------------------------
# 3. 核心对话循环
# -------------------------------------------------------------
def run_single_dialogue(dialogue_id: str, patient_data: Dict, doctor_agent: DoctorAgent, 
                        patient_client: OpenAI, patient_model_name: str, patient_style: str):
    """
    运行一次完整的医生-患者多轮对话，引入 patient_style 参数。
    """
    print(f"\n--- 开始对话 #{dialogue_id} ({patient_style}风格) ---")
    
    # 1. 初始化患者模拟器 Prompt
    patient_system_prompts = [
        f"你现在扮演一位正在寻求咨询的心理问题患者，风格是【{patient_style}】。你的档案是：{patient_data['user_info']['full_instruction']}。请严格基于你的档案回复，保持简洁。每一轮回复尽量控制在 20 字以内，或者 1-2 句话。 永远不要主动开启新话题。只有当医生明确询问时，才回答相关信息,只给一些表层线索,不要发散症状，让医生挖掘",
        f"你现在扮演一位正在寻求咨询的心理问题患者，风格是【{patient_style}】。你的档案是：{patient_data['user_info']['full_instruction']}。请严格基于你的档案回复，正常交流，能够清晰描述自己的症状，会适当配合医生的询问提供细节，但不会长篇大论。",
        f"你现在扮演一位正在寻求咨询的心理问题患者，风格是【{patient_style}】。你的档案是：{patient_data['user_info']['full_instruction']}。请严格基于你的档案回复，你渴望倾诉。回答问题时可以适当发散，主动提供一些额外的细节，甚至包含一些无关的抱怨。",
    
    ]
    if patient_style == "内向":
        patient_system_prompt=patient_system_prompts[0]
    elif patient_style== "中性":
        patient_system_prompt=patient_system_prompts[1]
    elif patient_style== "健谈":
        patient_system_prompt=patient_system_prompts[2]
        
    
    doctor_agent.reset_state()
    dialogue_log = []
    turn_counter = 0
    
    termination_reason = "MAX_TURNS_REACHED"

    # =========================================================================
    # 
    # Doctor Agent 的 Prompt 模板在 doctor_agent.py 中，它要求输入一个 user_input。
    # 在第一轮，我们使用 patient_data['user_info']['full_instruction'] 作为首个输入。
    #
    # 注意：我们将 patient_data['user_info']['full_instruction'] 视为 Doctor Agent 接收到的
    # “第一个来自患者或家属的描述”，并用它来初始化对话。
    # =========================================================================
    
    # 将患者的基本信息作为 Doctor Agent 的首个 User Input
    initial_user_input = patient_data['user_info']['full_instruction']

    # ------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    # 第一轮 Doctor Agent 的输入
    user_input = initial_user_input 

    while turn_counter < MAX_TURNS:
        turn_counter += 1
        
        try:
            # Step 1: Doctor Agent 执行一轮 (Predict, Prompt, API Call)
            # Doctor Agent 会接收 user_input，生成回复，并更新其内部 history
            doctor_step_result = doctor_agent.step(user_input)
            doctor_response = doctor_step_result['response']
            
            if doctor_step_result['actual_topic'] == "结束语":
                termination_reason = "AGENT_TERMINATED_CONFIRMED"
                
            # Step 2: 记录本轮数据 (在 Doctor 回复之后记录)
            turn_data = {
                "turn": turn_counter,
                "user_input": user_input, # 本轮 Doctor Agent 处理的输入
                "doctor_response": doctor_response,
                "bert_predicted_topic": doctor_step_result['predicted_topic'],
                "sft_actual_topic": doctor_step_result['actual_topic'],
                "topic_history_state": list(doctor_agent.topic_history)
            }
            dialogue_log.append(turn_data)
            
            print(f"[T{turn_counter} | {patient_style} | {doctor_step_result['actual_topic']}] D: {doctor_response[:30]}... | P: {user_input[:20]}...")
            
            if termination_reason != "MAX_TURNS_REACHED":
                break
                
            # Step 3: 患者 Agent 回复 (生成下一轮的 user_input)
            # 传入包含 [P1, D1, P2, D2, ..., P(n)] 的最新历史
            next_user_input = actual_patient_llm_call(
                patient_system_prompt, 
                doctor_response, 
                doctor_agent.dialogue_history, # 传入 Doctor Agent 更新后的 history
                patient_client, 
                patient_model_name
            )
            
            user_input = next_user_input # 更新下一轮 Doctor Agent 的输入

        except Exception as e:
            print(f"对话 #{dialogue_id} 在第 {turn_counter} 轮出错: {e}")
            termination_reason = "RUNTIME_ERROR"
            break

    # 5. 保存结果 (保持不变)
    final_result = {
        "dialogue_id": dialogue_id,
        "patient_style": patient_style,
        "patient_info": patient_data['user_info']['full_instruction'],
        "patient_target_topics": patient_data['topics'],
        "dialogue_history": dialogue_log,
        "termination_reason": termination_reason
    }
    
    file_path = os.path.join(LOG_DIR, f"dialogue_{dialogue_id}_{patient_style}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)
        
    print(f"--- 对话 #{dialogue_id} ({patient_style}) 结束, 原因: {termination_reason}。总轮次: {turn_counter}。---")

def main():
    """主函数：加载配置并运行所有评估。"""
    
    # 1. 检查和创建日志目录
    ensure_directory(LOG_DIR)
    
    # 2. 硬件配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 3. 话题预测器实例化
    try:
        topic_predictor = TopicPredictor(
            model_path=TOPIC_PREDICTOR_CONFIG['bert_path'], 
            transition_matrix_path=TOPIC_PREDICTOR_CONFIG['matrix_path'], 
            device=DEVICE,
            alpha=1.0, 
            max_length=512
        )
    except FileNotFoundError as e:
        print(f"错误: 话题预测器初始化失败。请检查路径。")
        print(e)
        return

    # 4. Doctor Agent 实例化
    doctor_agent = DoctorAgent(
        topic_predictor=topic_predictor, 
        model_name=SFT_MODEL_CONFIG['model_name'],
        api_key=SFT_MODEL_CONFIG['api_key'],
        base_url=SFT_MODEL_CONFIG['base_url']
    )

    # 5. Patient Agent 客户端配置
    patient_client = OpenAI(api_key=PATIENT_LLM_CONFIG['api_key'], base_url=PATIENT_LLM_CONFIG['base_url']) 
    patient_model_name = PATIENT_LLM_CONFIG['model_name']

    # 6. 加载患者数据
    try:
        patient_data_list = load_patient_data(PATIENT_DATA_FILE)
    except FileNotFoundError as e:
        print(f"错误: 患者数据文件未找到。请修改 PATIENT_DATA_FILE 路径。")
        return

    # 7. 运行所有对话 (实现风格轮流选择)
    print(f"成功加载 {len(patient_data_list)} 条患者数据。开始评估...")
    
    dialogue_counter = 0
    
    for data in patient_data_list:
        base_id = data.get('user_id', dialogue_counter + 1)
        
        selected_style = random.choice(PATIENT_STYLES)
        dialogue_id = f"{base_id}_R{dialogue_counter + 1}"
            
        run_single_dialogue(
            dialogue_id=dialogue_id, 
            patient_data=data, 
            doctor_agent=doctor_agent, 
            patient_client=patient_client,
            patient_model_name=patient_model_name,
            patient_style=selected_style
        )
        time.sleep(1)
        dialogue_counter += 1


if __name__ == "__main__":
    main()
