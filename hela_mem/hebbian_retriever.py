from .utils import gpt_generate_answer

class HebbianRetriever:
    def __init__(self, memory_graph, profile_memory=None):
        self.graph = memory_graph
        # Optional: Connect to a static profile memory if we want strictly separate semantic layer
        # But in pure Hebbian philosophy, the graph contains everything.
        self.profile_memory = profile_memory

    def answer(self, query, speaker_a="User", speaker_b="AI", top_k=5, knowledge_top_k=10):
        """
        Generate answer using Hebbian retrieval context
        """
        # 1. Retrieve from Graph
        # This handles Vector Search + Spreading Activation + Reinforcement internally
        results = self.graph.retrieve(query, top_k=top_k)
        
        # 2. Retrieve from Knowledge Base (Semantic Memory) with enhanced query
        long_knowledge = []
        if self.profile_memory and hasattr(self.profile_memory, 'search_knowledge'):
            try:
                long_knowledge = self.profile_memory.search_knowledge(query, top_k=knowledge_top_k)
            except Exception as e:
                print(f"Warning: Could not retrieve knowledge base: {e}")
        
        # 3. Format Context
        context_blocks = []
        for res in results:
            node = res["node"]
            score = res["score"]
            
            # Format block
            source_label = "Direct Match" if res["base_score"] > 0.6 else "Associative Memory"
            block = (
                f"[{source_label} | Relevancy: {score:.2f}]\n"
                f"Time: {node['timestamp']}\n"
                f"Content: {node['content']}"
            )
            context_blocks.append(block)
        context_text = "\n\n".join(context_blocks)
        
        # 4. Get Profile
        profile_text = "None"
        if self.profile_memory:
             if hasattr(self.profile_memory, 'get_raw_user_profile'):
                 profile_text = self.profile_memory.get_raw_user_profile(self.graph.file_path.split('/')[-1].split('_')[0])
             elif isinstance(self.profile_memory, dict):
                 profile_text = str(self.profile_memory.get("data", "None"))
        
        # 5. Get Assistant Knowledge
        assistant_knowledge_text = ""
        if self.profile_memory and hasattr(self.profile_memory, 'get_assistant_knowledge'):
            try:
                ak_list = self.profile_memory.get_assistant_knowledge()
                if ak_list:
                    assistant_knowledge_text = "Here are some of your character traits and knowledge:\n"
                    for item in ak_list:
                        k_text = item['knowledge'].strip()
                        assistant_knowledge_text += f"- {k_text}\n"
                    assistant_knowledge_text += "\n"
            except Exception as e:
                print(f"Warning: Could not retrieve assistant knowledge: {e}")
        
        # 6. Build System Prompt
        system_prompt = (
            f"You are role-playing as {speaker_b} in a conversation with the user is playing is {speaker_a}. "
            f"{assistant_knowledge_text}"
            f"Any content referring to 'User' in the prompt refers to {speaker_a}'s content, and any content referring to 'AI' or 'assistant' refers to {speaker_b}'s content.\n"
            f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
            f"When the question is: \"What did the charity race raise awareness for?\", you should not answer in the form of: \"The charity race raised awareness for mental health.\" Instead, it should be: \"mental health\", as this is more concise."
        )
        
        # 7. Build Knowledge Base text
        knowledge_text = ""
        if long_knowledge:
            knowledge_text = "<KNOWLEDGE BASE>\n"
            for kn in long_knowledge:
                knowledge_text += f"- {kn['knowledge']}\n"
            knowledge_text += "\n"
        
        # 8. Build User Prompt
        user_prompt = (
            f"<CONTEXT>\n"
            f"Relevant memories:\n"
            f"{context_text}\n\n"
            f"{knowledge_text}"
            f"<CHARACTER TRAITS>\n"
            f"Characteristics of {speaker_a}:\n"
            f"{profile_text}\n\n"
            f"The question is: {query}\n"
            f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
            f"Please only provide the content of the answer, without including 'answer:'\n"
            f"For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.\n"
            f"If the question is about the duration, answer in the form of several years, months, or days.\n"
            f"Generate answers primarily composed of concrete entities, such as Mentoring program, school speech, etc"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 9. Generate
        response = gpt_generate_answer(user_prompt, messages)

        return response, results

    def process_conversation_turn(self, user_input, agent_response, timestamp=None):
        """
        Process a turn to add it to memory.
        This is the 'Encoding' phase.
        """
        # Combine into a single episodic chunk
        content = f"User: {user_input}\nAI: {agent_response}"
        self.graph.add_memory(content, role="interaction", timestamp=timestamp)
