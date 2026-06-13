from llama_index.core.agent.workflow import ReActAgent
from tools import load_model_tool, model_data_tool, model_info_tool, current_model_tool, check_load_model_tool
from tools import reaction_info_tool, metabolite_info_tool, gene_info_tool
from tools import run_fba_tool, set_objective_tool, run_fva_tool
from tools import gene_knockout_tool, reaction_knockout_tool, flux_sampler_tool
from tools import list_escher_maps_tool, visualize_escher_tool, memote_report_tool
from tools import run_pfba_tool, run_geometric_fba_tool, add_reaction_tool, set_reaction_bounds_tool, build_model_with_carveme_tool, build_context_model_with_corda_tool
from tools import (
    find_essential_genes_tool, find_essential_reactions_tool, check_model_consistency_tool,
    check_mass_balance_tool, prune_unused_reactions_tool, prune_unused_metabolites_tool,
    find_minimal_medium_tool, build_model_with_mackinac_tool,
)
from tools import request_file_upload_tool
from llama_index.core.llms import ChatMessage
from prompts import system_prompt, agent_context, llm_system_prompt, llm_prompt, chat_file_upload_context, chat_credential_context
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
    build_model_with_carveme_tool,
    build_context_model_with_corda_tool,
    find_essential_genes_tool,
    find_essential_reactions_tool,
    check_model_consistency_tool,
    check_mass_balance_tool,
    prune_unused_reactions_tool,
    prune_unused_metabolites_tool,
    find_minimal_medium_tool,
    build_model_with_mackinac_tool,
    request_file_upload_tool,
]

def setup_agent(new_llm):
    global agent, llm
    llm = new_llm
    agent = ReActAgent(
        tools=all_tools,
        llm=llm,
        system_prompt=system_prompt + "\n\n" + agent_context + "\n\n" + chat_file_upload_context + "\n\n" + chat_credential_context,
        verbose=True,
    )

async def agent_query(user_input: str):
    handler = agent.run(user_input)
    agent_response = await handler

    # Detect if the agent called request_file_upload this turn.
    # agent_response.tool_calls is a list[ToolSelection] with .tool_name and .tool_kwargs.
    for tc in agent_response.tool_calls:
        if getattr(tc, "tool_name", None) == "request_file_upload":
            kw = tc.tool_kwargs
            file_types_raw = kw.get("file_types", "")
            file_types = (
                [ft.strip() for ft in file_types_raw.split(",") if ft.strip()]
                if isinstance(file_types_raw, str)
                else list(file_types_raw)
            )
            return {
                "__upload_required__": True,
                "param": kw.get("param", "file"),
                "file_types": file_types,
                "description": kw.get("description", "Please upload the required file."),
            }

    # Normal path — second LLM reformatting call
    final_prompt = llm_prompt.replace("<user_input>", user_input)
    final_prompt = final_prompt.replace("<agentResponse>", str(agent_response.response.content))
    messages = [
        ChatMessage(role="system", content=llm_system_prompt),
        ChatMessage(role="user", content=final_prompt.strip())
    ]
    response = await llm.achat(messages)
    return str(response.message.content)