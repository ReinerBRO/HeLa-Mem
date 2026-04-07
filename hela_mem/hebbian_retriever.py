import os
from .utils import gpt_generate_answer
from .reranker import rerank_memories

class HebbianRetriever:
    def __init__(self, memory_graph, profile_memory=None, use_planner=False, use_investigator=False, use_critic=False, use_surgeon=False, use_architect=False, use_hippocampus=False, use_extra_prompt=False):
        self.graph = memory_graph
        # Optional: Connect to a static profile memory if we want strictly separate semantic layer
        # But in pure Hebbian philosophy, the graph contains everything.
        self.profile_memory = profile_memory 
        self.use_planner = use_planner
        self.use_investigator = use_investigator
        self.use_critic = use_critic
        self.use_surgeon = use_surgeon
        self.use_architect = use_architect
        self.use_hippocampus = use_hippocampus
        self.use_extra_prompt = use_extra_prompt  # [NEW] Separate Hebbian memories in prompt

    def answer(self, query, speaker_a="User", speaker_b="AI", top_k=5, knowledge_top_k=10):
        """
        Generate answer using Hebbian retrieval context
        """
        # [NEW] Planner Agent Interface
        search_query = query
        if self.use_planner:
            search_query = self.planner_agent(query)

        # 1. Retrieve from Graph
        # This handles Vector Search + Spreading Activation + Reinforcement internally
        # If Hippocampus is ON, we retrieve MORE (Expansion Phase)
        # If Reranker is ON, we also retrieve more for reranking pool
        use_reranker = os.environ.get('HEBBIAN_USE_RERANKER', 'false').lower() == 'true'
        rerank_pool_multiplier = int(os.environ.get('HEBBIAN_RERANK_POOL_MULTIPLIER', '3'))
        
        if self.use_hippocampus:
            retrieve_k = top_k * 3
        elif use_reranker:
            retrieve_k = top_k * rerank_pool_multiplier
        else:
            retrieve_k = top_k
            
        results = self.graph.retrieve(search_query, top_k=retrieve_k)
        
        # [NEW] Reranker Phase - rerank before any other processing
        if use_reranker and len(results) > top_k:
            # Convert results to memory format for reranking
            memories_to_rerank = [{"content": r["node"]["content"], **r} for r in results]
            reranked = rerank_memories(query, memories_to_rerank, top_k=top_k if not self.use_hippocampus else retrieve_k)
            # Reconstruct results from reranked memories
            results = [{"node": r["node"], "score": r["score"], "base_score": r["base_score"], "source": r.get("source", "base")} for r in reranked]
        
        # [NEW] Hippocampal Selection Phase
        if self.use_hippocampus:
            # Select Top-K (e.g. 15) from the expanded pool (e.g. 45 or 90)
            # We pass the original top_k as the target count
            results = self.hippocampal_agent(query, results, target_count=top_k)
        
        # [DISABLED] Cross-Memory Spreading: Extract keywords from Hebbian-flipped entries
        # These keywords will prime the Semantic Memory (KB) retrieval
        # hebbian_keywords = []
        # for res in results:
        #     if res.get("source") == "hebbian":  # Only from Hebbian-flipped entries
        #         node_keywords = res["node"].get("keywords", [])
        #         hebbian_keywords.extend(node_keywords)
        
        # Build enhanced KB query - DISABLED, use original query
        # if hebbian_keywords:
        #     # Deduplicate and limit
        #     unique_keywords = list(dict.fromkeys(hebbian_keywords))[:3]
        #     enhanced_kb_query = search_query + " " + " ".join(unique_keywords)
        #     print(f"  [Cross-Memory] KB query enhanced with {len(unique_keywords)} keywords from Hebbian activation.")
        # else:
        #     enhanced_kb_query = search_query
        enhanced_kb_query = search_query  # Use original query directly
        
        # 2. Retrieve from Knowledge Base (Semantic Memory) with enhanced query
        long_knowledge = []
        if self.profile_memory and hasattr(self.profile_memory, 'search_knowledge'):
            try:
                long_knowledge = self.profile_memory.search_knowledge(enhanced_kb_query, top_k=knowledge_top_k)
            except Exception as e:
                print(f"Warning: Could not retrieve knowledge base: {e}")
        
        # 3. Format Context - separate base and hebbian if extra prompt mode
        base_blocks = []
        hebbian_blocks = []
        
        for res in results:
            node = res["node"]
            score = res["score"]
            source_type = res.get("source", "base")
            
            # Format block
            source_label = "Direct Match" if res["base_score"] > 0.6 else "Associative Memory"
            if "Hippocampus" in str(res.get("source", "")): source_label = "Hippocampal Selection"
            
            block = (
                f"[{source_label} | Relevancy: {score:.2f}]\n"
                f"Time: {node['timestamp']}\n"
                f"Content: {node['content']}"
            )
            
            # Separate by source type
            if source_type == "hebbian":
                hebbian_blocks.append(block)
            else:
                base_blocks.append(block)
        
        # Build context text based on mode
        if self.use_extra_prompt and hebbian_blocks:
            # Separate sections for base and hebbian
            base_text = "\n\n".join(base_blocks)
            hebbian_text = "\n\n".join(hebbian_blocks)
            context_text = f"<RETRIEVED MEMORIES>\n{base_text}\n\n<ASSOCIATED MEMORIES (via Hebbian Spreading)>\n{hebbian_text}"
        else:
            # Original: all mixed together
            context_blocks = base_blocks + hebbian_blocks
            context_text = "\n\n".join(context_blocks)
        
        # 4. Get Profile (can be disabled via env)
        profile_text = "None"
        use_profile = os.environ.get('HEBBIAN_USE_PROFILE', 'true').lower() == 'true'
        if use_profile and self.profile_memory:
             if hasattr(self.profile_memory, 'get_raw_user_profile'):
                 profile_text = self.profile_memory.get_raw_user_profile(self.graph.file_path.split('/')[-1].split('_')[0])
             elif isinstance(self.profile_memory, dict):
                 profile_text = str(self.profile_memory.get("data", "None"))
        
        # 5. Get Assistant Knowledge (can be disabled via env)
        assistant_knowledge_text = ""
        use_assistant_knowledge = os.environ.get('HEBBIAN_USE_ASSISTANT_KNOWLEDGE', 'true').lower() == 'true'
        if use_assistant_knowledge and self.profile_memory and hasattr(self.profile_memory, 'get_assistant_knowledge'):
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
        # [ARCHITECT FUSION]
        if self.use_architect:
             user_prompt = (
                f"<ROLE>\n"
                f"You are an expert biographer and data analyst.\n\n"
                f"<BACKGROUND STORY> (High-level Context)\n"
                f"{knowledge_text}\n"
                f"(Use this to understand the timeline and main events.)\n\n"
                f"<EVIDENCE FRAGMENTS> (Specific Details)\n"
                f"{context_text}\n"
                f"(Use these to find specific dates, names, and entities.)\n\n"
                f"<CHARACTER TRAITS>\n"
                f"{profile_text}\n\n"
                f"<TASK>\n"
                f"Question: {query}\n"
                f"Answer the question by FUSING the Background Story with the Evidence Fragments.\n"
                f"1. If the Background Story gives the general answer, use Evidence Fragments to make it specific (e.g., add exact date).\n"
                f"2. If Evidence Fragments conflict with Background Story, trust the specific Evidence Fragments.\n"
                f"3. Be extremely concise. Do not include 'answer:'.\n"
                f"4. For dates, use format '15 July 2023'.\n"
            )
        else:
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
        
        # [NEW] Investigator Agent Interface (Iterative Retrieval)
        if self.use_investigator:
            new_query = self.investigator_agent(query, response)
            if new_query:
                # Retrieve additional context
                extra_results = self.graph.retrieve(new_query, top_k=top_k)
                # Merge results (avoid duplicates)
                existing_ids = {res["node"]["id"] for res in results}
                for res in extra_results:
                    if res["node"]["id"] not in existing_ids:
                        results.append(res)
                        # Add to context text
                        node = res["node"]
                        score = res["score"]
                        source = "Investigator Search"
                        block = (
                            f"[{source} | Relevancy: {score:.2f}]\n"
                            f"Time: {node['timestamp']}\n"
                            f"Content: {node['content']}"
                        )
                        context_blocks.append(block)
                
                # Re-build Context and Prompt (Simplified for brevity, ideally loop back)
                context_text = "\n\n".join(context_blocks)
                # ... (Re-prompt logic would go here)

        # [NEW] Critic Agent Interface
        if self.use_critic:
            response = self.critic_agent(query, response)
        
        return response, results

    def hippocampal_agent(self, query, results, target_count=15):
        """
        Hippocampal Selection Agent.
        Selects the most relevant `target_count` memories from a larger pool.
        """
        if not results or len(results) <= target_count:
            return results
            
        try:
            # Format candidates for the LLM
            candidates_text = ""
            for i, res in enumerate(results):
                # Truncate content for token efficiency
                content_preview = res['node']['content'][:200] + "..." if len(res['node']['content']) > 200 else res['node']['content']
                candidates_text += f"[{i}] {content_preview}\n"
            
            hip_system_prompt = (
                "You are the Hippocampal Selector.\n"
                "Your task is to select the most relevant memories for a given query.\n"
                "RULES:\n"
                "1. Read the User Question and the Candidate Memories.\n"
                f"2. Select exactly {target_count} memories that best answer the question.\n"
                "3. Prioritize memories that contain specific entities (names, dates, locations) relevant to the query.\n"
                "4. Return the INDICES of the selected memories as a JSON list (e.g., [0, 2, 4]).\n"
                "5. Output ONLY the JSON list."
            )
            
            hip_user_prompt = f"User Question: {query}\n\nCandidate Memories:\n{candidates_text}\n\nSelected Indices (JSON):"
            
            messages = [
                {"role": "system", "content": hip_system_prompt},
                {"role": "user", "content": hip_user_prompt}
            ]
            
            response_text = gpt_generate_answer(hip_user_prompt, messages).strip()
            
            # Parse JSON list
            import json
            try:
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                if start != -1 and end != -1:
                    json_str = response_text[start:end]
                    indices = json.loads(json_str)
                else:
                    indices = [int(x.strip()) for x in response_text.split(',') if x.strip().isdigit()]
            except:
                print(f"Hippocampal Agent failed to parse JSON: {response_text}")
                return results[:target_count] # Fallback
                
            # Filter results
            selected_results = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(results):
                    # Mark source
                    res = results[idx]
                    res["source"] = "Hippocampal Selection"
                    selected_results.append(res)
            
            # Ensure we have enough (if LLM returned too few, pad with top base results)
            if len(selected_results) < target_count:
                existing_ids = {res["node"]["id"] for res in selected_results}
                for res in results:
                    if len(selected_results) >= target_count: break
                    if res["node"]["id"] not in existing_ids:
                        selected_results.append(res)
            
            # Ensure we don't have too many
            selected_results = selected_results[:target_count]
            
            print(f"[Hippocampus] Selected {len(results)} -> {len(selected_results)} memories.")
            return selected_results
            
        except Exception as e:
            print(f"Hippocampal Agent failed: {e}")
            return results[:target_count]

    def surgeon_agent(self, query, results):
        """
        External Surgeon Agent to refine context by removing noise.
        Returns the filtered list of result objects.
        """
        if not results:
            return []
            
        try:
            # Format candidates for the LLM
            candidates_text = ""
            for i, res in enumerate(results):
                candidates_text += f"[{i}] {res['node']['content']}\n"
            
            surgeon_system_prompt = (
                "You are a Context Surgeon. Your task is to filter a list of retrieved memories.\n"
                "RULES:\n"
                "1. Read the User Question and the Candidate Memories.\n"
                "2. Identify which memories are RELEVANT to answering the question.\n"
                "3. Return the INDICES of the relevant memories as a JSON list (e.g., [0, 2, 4]).\n"
                "4. If a memory is noise or completely irrelevant, exclude its index.\n"
                "5. If NO memories are relevant, return an empty list [].\n"
                "6. Output ONLY the JSON list."
            )
            
            surgeon_user_prompt = f"User Question: {query}\n\nCandidate Memories:\n{candidates_text}\n\nRelevant Indices (JSON):"
            
            messages = [
                {"role": "system", "content": surgeon_system_prompt},
                {"role": "user", "content": surgeon_user_prompt}
            ]
            
            response_text = gpt_generate_answer(surgeon_user_prompt, messages).strip()
            
            # Parse JSON list
            import json
            try:
                # Try to find JSON-like structure if there's extra text
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                if start != -1 and end != -1:
                    json_str = response_text[start:end]
                    indices = json.loads(json_str)
                else:
                    # Fallback: try to parse comma separated numbers
                    indices = [int(x.strip()) for x in response_text.split(',') if x.strip().isdigit()]
            except:
                print(f"Surgeon Agent failed to parse JSON: {response_text}")
                return results # Fallback to keep all
                
            # Filter results
            filtered_results = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(results):
                    filtered_results.append(results[idx])
            
            print(f"[Surgeon] Filtered {len(results)} -> {len(filtered_results)} memories.")
            return filtered_results
            
        except Exception as e:
            print(f"Surgeon Agent failed: {e}")
            return results

    def critic_agent(self, query, initial_answer):
        """
        External Critic Agent to refine the answer for metric optimization.
        This acts as a post-processing step to enforce formatting rules.
        """
        try:
            critic_system_prompt = (
                "You are a strict editor optimizing answers for an automated evaluation metric (F1/BLEU).\n"
                "Your goal is to make the answer extremely concise and strictly formatted.\n"
                "RULES:\n"
                "1. Remove leading articles ('The', 'A', 'An').\n"
                "2. Remove trailing punctuation ('.').\n"
                "3. Format dates strictly as '15 July, 2023' (Day Month, Year).\n"
                "4. Keep ONLY the core entity or key phrase.\n"
                "5. If the answer is already perfect, output it unchanged."
            )
            
            critic_user_prompt = (
                f"Original Question: {query}\n"
                f"Initial Answer: {initial_answer}\n\n"
                f"Refined Answer (content only):"
            )
            
            messages = [
                {"role": "system", "content": critic_system_prompt},
                {"role": "user", "content": critic_user_prompt}
            ]
            
            refined_answer = gpt_generate_answer(critic_user_prompt, messages)
            return refined_answer.strip()
            
        except Exception as e:
            print(f"Critic Agent failed: {e}")
            return initial_answer

    def investigator_agent(self, query, current_answer):
        """
        External Investigator Agent to check answer sufficiency.
        Returns 'SUFFICIENT' or a new search query.
        """
        try:
            inv_system_prompt = (
                "You are a Quality Assurance Investigator for a memory system.\n"
                "Your task is to check if the Current Answer fully and specifically answers the User Question.\n"
                "RULES:\n"
                "1. If the answer is VAGUE (e.g., 'a friend', 'somewhere', 'he did it') but lacks specific names/dates/places, output the SPECIFIC KEYWORDS needed to find the missing info (e.g., 'Jon friend name').\n"
                "2. If the answer misses part of the question (e.g., asks for 'Who and When' but only answers 'Who'), output a QUERY for the missing part.\n"
                "3. If the answer is complete and contains concrete entities, output 'SUFFICIENT'.\n"
                "4. Output ONLY the query string or 'SUFFICIENT'. Do NOT output the text 'NEW KEYWORD QUERY'."
            )
            
            inv_user_prompt = f"User Question: {query}\nCurrent Answer: {current_answer}\nDecision:"
            
            messages = [
                {"role": "system", "content": inv_system_prompt},
                {"role": "user", "content": inv_user_prompt}
            ]
            
            decision = gpt_generate_answer(inv_user_prompt, messages).strip()
            
            if "SUFFICIENT" in decision.upper() or len(decision) < 3:
                return None
            
            print(f"[Investigator] Insufficient. New Search: '{decision}'")
            return decision
            
        except Exception as e:
            print(f"Investigator Agent failed: {e}")
            return None

    def planner_agent(self, query):
        """
        External Planner Agent to decompose or expand complex queries.
        This acts as a pre-processing step to improve retrieval recall.
        """
        try:
            planner_system_prompt = (
                "You are a Query Optimizer for a Hebbian Memory Graph.\n"
                "Your task is to rewrite the user's question into a keyword-rich search query.\n"
                "RULES:\n"
                "1. If the question is simple, return it unchanged.\n"
                "2. If the question is Multi-hop (e.g. 'Who is the boss of the person who...'), rewrite it to include keywords for ALL hops.\n"
                "3. If the question implies a time range, add specific temporal keywords if possible.\n"
                "4. Output ONLY the rewritten query."
            )
            
            planner_user_prompt = f"Original Question: {query}\nOptimized Query:"
            
            messages = [
                {"role": "system", "content": planner_system_prompt},
                {"role": "user", "content": planner_user_prompt}
            ]
            
            optimized_query = gpt_generate_answer(planner_user_prompt, messages)
            # If the optimizer fails or returns empty, fallback to original
            if not optimized_query or len(optimized_query) < 5:
                return query
                
            print(f"[Planner] Optimized: '{query}' -> '{optimized_query}'")
            return optimized_query.strip()
            
        except Exception as e:
            print(f"Planner Agent failed: {e}")
            return query

    def process_conversation_turn(self, user_input, agent_response, timestamp=None):
        """
        Process a turn to add it to memory.
        This is the 'Encoding' phase.
        """
        # Combine into a single episodic chunk
        content = f"User: {user_input}\nAI: {agent_response}"
        self.graph.add_memory(content, role="interaction", timestamp=timestamp)

