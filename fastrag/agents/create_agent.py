import torch
import yaml
from transformers import AutoTokenizer, StoppingCriteriaList
from typing import Dict, Any, Optional

from fastrag.agents.base import Agent, ToolsManager
from fastrag.agents.memory.conversation_memory import ConversationMemory
from fastrag.agents.tools.tools import TOOLS_FACTORY
from fastrag.generators.stopping_criteria.stop_words import StopWordsByTextCriteria


DEBUG_MODE = True


AGENT_SYSTEM_ROLES = [
    {
        "role": "system",
        "content": """You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

{custom_system_instructions}

## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

Previous conversation context:
{conversation_history}

You have access to the following tools:
{tool_names_with_descriptions}

## Output Format

If you lack information to answer, you MUST use a tool that can help you to get more information to answer the question, and use MUST use the following format:

```
Thought: I should use a tool to help me answer the question.
Tool: [tool name if using a tool].
Tool Input: [the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world"}})].
Observation: [tool response]
```

If you have enough information to answer the question without using any more tools, you MUST finish with "Final Answer:" and respond in the following format:

```
Thought: I can answer without using any more tools.
Final Answer: [your answer here]
```
""",
    }
]


AGENT_CONVERSATION_BASE_ROLES = [
    {
        "role": "user",
        "content": """{query}
Context: {context}
Previous tools used: {tool_history}
Thought: """,
    },
]

AGENT_ROLES = {"system": AGENT_SYSTEM_ROLES, "chat": AGENT_CONVERSATION_BASE_ROLES}

class EnhancedConversationMemory(ConversationMemory):
    def __init__(self, generator):
        super().__init__(generator)
        self.tool_history = []
        self.user_data = {}
        self.system_configs = {}

    def add_interaction(self, query: str, response: str, metadata: Optional[Dict] = None):
        if DEBUG_MODE:
            print(f"Query: {query}")
            print(f"Metadata: {metadata}")
            
        self.tool_history.append({
            "query": query,
            "response": response,
            "metadata": metadata,
            "system_config": self.system_configs 
        })

def get_generator(chat_model_config: Dict[str, Any]):

    class_path = chat_model_config["generator_class"]
    class_path_parts = class_path.split(".")
    current_module = __import__(class_path_parts[0])
    for part in class_path_parts[1:]:
        current_module = getattr(current_module, part)
    generator_class = current_module

    tokenizer = AutoTokenizer.from_pretrained(chat_model_config["generator_kwargs"]["model"])
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    stop_word_list = ["Observation:", "<|eot_id|>", "<|end|>"]
    sw = StopWordsByTextCriteria(tokenizer=tokenizer, stop_words=stop_word_list, device="cpu")
    stopping_criteria_list = StoppingCriteriaList([sw])

  
    if DEBUG_MODE:
        print(f"Model config: {chat_model_config}")
        
    generator = generator_class(**chat_model_config["generator_kwargs"])
    generator.warm_up()

    return generator, tokenizer

class EnhancedToolsManager(ToolsManager):
    def __init__(self, tools, debug_mode=False):
        super().__init__(tools)
        self.debug_mode = debug_mode
    
        self.tool_execution_history = []

    def execute_tool(self, tool_name: str, **kwargs):
        try:
            result = super().execute_tool(tool_name, **kwargs)
    
            self.tool_execution_history.append({
                "tool": tool_name,
                "inputs": kwargs,
                "result": result,
                "status": "success"
            })
            return result
        except Exception as e:
            if self.debug_mode:
                print(f"Tool execution error: {str(e)}")
                print(f"Tool inputs: {kwargs}")
            raise

def get_basic_conversation_pipeline(args):

    conversation_config = yaml.safe_load(open(args.config, "r"))
    

    if DEBUG_MODE:
        print(f"Loaded config: {conversation_config}")

    chat_model_config = conversation_config["chat_model"]
    generator, tokenizer = get_generator(chat_model_config)

    tools_objects_map = {}
    if "tools" in conversation_config:
        for tool_config in conversation_config["tools"]:
      
            tool_type = tool_config["type"]
            tool_type_class = TOOLS_FACTORY[tool_type]
            
   
            tool_obj = tool_type_class(**tool_config["params"])
            tools_objects_map[tool_config["params"]["name"]] = tool_obj

    return generator, tokenizer, tools_objects_map

def get_agent_conversation_pipeline(args, custom_instructions=None):
    generator, tokenizer, tools_objects_map = get_basic_conversation_pipeline(args)


    if custom_instructions:
        AGENT_SYSTEM_ROLES[0]["content"] = AGENT_SYSTEM_ROLES[0]["content"].format(
            custom_system_instructions=custom_instructions,
            conversation_history="{conversation_history}",
            tool_names_with_descriptions="{tool_names_with_descriptions}"
        )


    tools_manager = EnhancedToolsManager(list(tools_objects_map.values()), debug_mode=DEBUG_MODE)

 
    memory = EnhancedConversationMemory(generator)
    

    conversational_agent = Agent(
        generator,
        prompt_template=AGENT_ROLES,
        memory=memory,
        tools_manager=tools_manager,
    )

    return conversational_agent
