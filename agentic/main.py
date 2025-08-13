import os
import argparse
import logging
from flow import create_main_flow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('coding_agent.log')
    ]
)

logger = logging.getLogger('main')

def main():
    """
    Run the coding agent to help with code operations
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Coding Agent - AI-powered coding assistant')
    parser.add_argument('--query', '-q', type=str, help='User query to process', required=False)
    parser.add_argument('--working-dir', '-w', type=str, default=os.path.join(os.getcwd(), "project"), 
                        help='Working directory for file operations (default: current directory)')
    parser.add_argument('--graphrag', '-g', type=bool, default=True, 
                        help='whether to use graph rag search while finding code snippets')
    parser.add_argument('--model', '-m', type=str, default="Salesforce/SFR-Embedding-Code-400M_R", 
                        help='Model to use for the sementic code search')
    args = parser.parse_args()
    
    # If no query provided via command line, ask for it
    user_query = args.query
    if not user_query:
        user_query = input("What would you like me to help you with? ")
    
    shared = {
        "user_query": user_query,
        "working_dir": args.working_dir,
        "history": [],
        "graphrag" : args.graphrag,
        "response": None
    }
    coding_agent_flow = create_main_flow(args.graphrag, args.model)
    while user_query:
    
        # Initialize shared memory
        shared["user_query"] = user_query
        
        logger.info(f"Working directory: {args.working_dir}")
        
        # Run the flow
        coding_agent_flow.run(shared)
        
        user_query = None
        if not user_query:
            user_query = input("What would you like me to help you with? ")
        

if __name__ == "__main__":
    main()