"""
RAG Agent: An agent-based framework for RAG that orchestrates retrieval and generation.
The agent decides which retrieval methods to use and how to combine results.
"""
from typing import List, Optional
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
import json
import time

class RAGAgent:
    """
    Agent-based RAG system that uses tools to perform retrieval and generation.
    """
    
    def __init__(self, llm, retrieval_tools, chunks: List[Document]):
        """
        Initialize RAG agent.
        
        Args:
            llm: LangChain LLM instance
            retrieval_tools: RetrievalTools instance
            chunks: List of all document chunks
        """
        self.llm = llm
        self.retrieval_tools = retrieval_tools
        self.chunks = chunks
        self.chunk_map = {
            (c.metadata.get("source"), c.metadata.get("chunk_number")): c 
            for c in chunks
        }
        
        # Create agent with tools
        tools = retrieval_tools.get_tools()
        self.agent = self._create_agent(llm, tools)
    
    def _create_agent(self, llm, tools):
        """Create LangChain agent with retrieval tools."""
        system_prompt = """You are an expert RAG (Retrieval-Augmented Generation) assistant.

Your task is to answer user questions by:
1. Understanding what information is needed
2. Selecting the appropriate retrieval method(s):
   - Use `dense_search_tool` for semantic/similarity queries
   - Use `bm25_search_tool` for exact keyword matching
   - Use `hybrid_search_tool` for comprehensive retrieval
   - Use `kg_search_tool` when query mentions entities, concepts, or relationships
   - Use `rerank_tool` to improve precision of results
3. Combining and analyzing retrieved information
4. Providing accurate, well-cited answers

Guidelines:
- Always use retrieval tools to find relevant information
- You can use multiple tools and combine their results
- Cite sources using chunk numbers when referencing information
- If information is not found, say so clearly
- Be concise but thorough in your answers
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return executor
    
    def query(self, user_query: str, chat_history: Optional[List] = None) -> dict:
        """
        Process a user query using the agent.
        
        Args:
            user_query: User's question
            chat_history: Optional chat history
        
        Returns:
            Dictionary with answer, citations, agent_steps, and timing info
        """
        # Run agent with timing
        t_start = time.time()
        result = self.agent.invoke({
            "input": user_query,
            "chat_history": chat_history or []
        })
        agent_time = time.time() - t_start
        
        # Extract answer
        answer = result.get("output", "")
        
        # Try to extract citations from agent's reasoning
        citations = self._extract_citations(result, answer)
        agent_steps = result.get("intermediate_steps", [])

        return {
            "answer": answer,
            "citations": citations,
            "agent_steps": agent_steps,
            "timing": {
                "agent_time": agent_time,
                "num_steps": len(agent_steps),
            },
        }
    
    def _extract_citations(self, result: dict, answer: str) -> List[dict]:
        """Extract citations from agent's tool usage."""
        citations = []
        seen_chunks = set()
        
        # Look through agent's tool calls
        for step in result.get("intermediate_steps", []):
            if len(step) >= 2:
                tool_output = step[1]
                if isinstance(tool_output, str):
                    try:
                        data = json.loads(tool_output)
                        if "results" in data:
                            for res in data["results"]:
                                source = res.get("source")
                                chunk_num = res.get("chunk_number")
                                page = res.get("page")
                                
                                if source and chunk_num:
                                    key = (source, chunk_num)
                                    if key not in seen_chunks:
                                        seen_chunks.add(key)
                                        chunk = self.chunk_map.get(key)
                                        if chunk:
                                            citations.append({
                                                "document": source,
                                                "page": page,
                                                "chunk_number": chunk_num,
                                                "chunk_content": chunk.page_content[:200]  # Preview
                                            })
                    except json.JSONDecodeError:
                        pass
        
        return citations

