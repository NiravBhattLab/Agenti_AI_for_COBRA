from llama_index.core.agent.workflow import ReActAgent
from tools import load_model_tool, model_data_tool, model_info_tool, current_model_tool, check_load_model_tool
from tools import reaction_info_tool, metabolite_info_tool, gene_info_tool
from tools import run_fba_tool, set_objective_tool, run_fva_tool
from tools import gene_knockout_tool, reaction_knockout_tool, flux_sampler_tool
from tools import list_escher_maps_tool, visualize_escher_tool, memote_report_tool
from tools import run_pfba_tool, run_geometric_fba_tool, add_reaction_tool, set_reaction_bounds_tool
from llama_index.core.llms import ChatMessage
from prompts import system_prompt, agent_context, llm_system_prompt, llm_prompt
from llama_index.core.memory import Memory
from dotenv import load_dotenv
import os

agent = None
llm = None
memory = Memory.from_defaults(session_id="metabolic_agent", token_limit=40000) # For future integration

all_tools = [
    load_model_tool, model_data_tool, model_info_tool, current_model_tool, check_load_model_tool,
    reaction_info_tool, metabolite_info_tool, gene_info_tool,
    run_fba_tool, set_objective_tool, run_fva_tool,
    gene_knockout_tool, reaction_knockout_tool, flux_sampler_tool,
    list_escher_maps_tool, visualize_escher_tool,
    memote_report_tool,
    run_pfba_tool, run_geometric_fba_tool,
    add_reaction_tool,
    set_reaction_bounds_tool,
]

def setup_agent(new_llm):
    global agent, llm
    llm = new_llm
    agent = ReActAgent(
        tools=all_tools,
        llm=llm,
        system_prompt=system_prompt + "\n\n" + agent_context,
        verbose=False,
    )

async def agent_query(user_input: str):
    handler = agent.run(user_input)
    agent_response = await handler
    final_prompt = llm_prompt.replace("<user_input>", user_input)
    final_prompt = final_prompt.replace("<agentResponse>", str(agent_response.response.content))
    messages = [
        ChatMessage(role="system", content=llm_system_prompt),
        ChatMessage(role="user", content=final_prompt.strip())
    ]
    response = await llm.achat(messages)
    return str(response.message.content)