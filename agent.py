from llama_index.core.agent import ReActAgent
from llama_index.core.tools import ToolMetadata
from tools import load_model_tool, model_data_tool, model_info_tool, current_model_tool, check_load_model_tool
from tools import reaction_info_tool, metabolite_info_tool, gene_info_tool
from tools import run_fba_tool, set_objective_tool, run_fva_tool
from tools import gene_knockout_tool, reaction_knockout_tool, flux_sampler_tool
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from prompts import system_prompt, agent_context, llm_system_prompt, llm_prompt
from llama_index.core.memory import Memory
import json

MODEL_NAME = "llama3.1:latest"
llm = Ollama(model=MODEL_NAME, request_timeout=300)
memory = Memory.from_defaults(session_id="metabolic_agent", token_limit=40000)

all_tools = [
    load_model_tool, model_data_tool, model_info_tool, # current_model_tool, check_load_model_tool,
    reaction_info_tool, metabolite_info_tool, gene_info_tool,
    run_fba_tool, set_objective_tool, run_fva_tool,
    gene_knockout_tool, reaction_knockout_tool, flux_sampler_tool
]

agent = ReActAgent.from_tools(
    tools=all_tools,
    llm=llm,
    system_prompt=system_prompt,
    context=agent_context,
    verbose=True
)

def setup_agent(new_llm):
    global agent, llm
    llm = new_llm
    agent = ReActAgent.from_tools(
        tools=all_tools,
        llm=llm,
        system_prompt=system_prompt,
        context=agent_context,
        verbose=True
    )

def agent_query(user_input: str):
    agent_response = agent.query(user_input)
    final_prompt = llm_prompt.replace("<user_input>", user_input)
    final_prompt = final_prompt.replace("<agentResponse>", str(agent_response))
    messages = [
        ChatMessage(role="system", content=llm_system_prompt),
        ChatMessage(role="user", content=final_prompt.strip())
    ]
    response = llm.chat(messages)
    return str(response.message.content)