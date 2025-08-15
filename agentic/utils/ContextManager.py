from utils.call_llm import call_llm
import logging

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages conversation context and summarization
    """
    
    def __init__(self):
        pass
    
    def summarize_conversation(self, query: str, response: str, past_summary: str = None) -> str:
        """
        Summarize the current conversation including the new query and response
        
        Args:
            query: The current user query
            response: The current response
            past_summary: Previous conversation summary (if any)
            
        Returns:
            str: Updated conversation summary
        """
        try:
            if past_summary is None:
                # First conversation, create initial summary
                prompt = f"""
Summarize the following conversation in a concise way that captures the key points:

User Query: {query}
Assistant Response: {response}

Please provide a brief summary that captures:
1. What the user asked for
2. What was accomplished
3. Key findings or changes made

Keep the summary concise but informative for future context.
"""
            else:
                # Add to existing summary
                prompt = f"""
Update the conversation summary with the new interaction:

Previous Summary: {past_summary}

New Interaction:
User Query: {query}
Assistant Response: {response}

Please provide an updated summary that:
1. Incorporates the new interaction
2. Maintains the key points from the previous summary
3. Keeps the overall summary concise and focused on the most important aspects

Focus on the progression of the conversation and what has been accomplished overall.
"""
            
            summary = call_llm(prompt)
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}")
            # Fallback to simple concatenation
            if past_summary is None:
                return f"User asked: {query}. Assistant provided response about the query."
            else:
                return f"{past_summary} | User asked: {query}. Assistant provided response."
