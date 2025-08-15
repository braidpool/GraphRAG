import os
import argparse
import logging
from pathlib import Path
from flow import create_main_flow
from utils.graph_creator import GraphBuilder
from utils.ContextManager import ContextManager

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

def validate_path_in_working_dir(path: str, working_dir: str) -> bool:
    """
    Validate that the given path is within the working directory
    """
    try:
        path_obj = Path(path).resolve()
        working_dir_obj = Path(working_dir).resolve()
        return path_obj.is_relative_to(working_dir_obj)
    except (ValueError, RuntimeError):
        return False

def create_graph_from_path(path: str, working_dir: str, model: str):
    """
    Create graph from the given path
    """
    if not validate_path_in_working_dir(path, working_dir):
        print(f"⚠️  Warning: The path '{path}' is not within the working directory '{working_dir}'")
        print("This file/folder will still be added to the graph, but it's not related to your current project.")
        response = input("Do you want to continue? (y/N): ")
        if response.lower() != 'y':
            return False
    
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"❌ Error: Path '{path}' does not exist")
            return False
        
        print(f"🔄 Creating graph from: {path}")
        print("This may take a while depending on the size of the codebase...")
        
        # Initialize GraphBuilder
        graph_builder = GraphBuilder()
        
        # Build graph from path
        result = graph_builder.build_graph_from_path(path_obj, is_dependency=False)
        
        if result.get("success", False):
            print(f"✅ Successfully created graph!")
            print(f"   Repository: {result.get('repository', 'Unknown')}")
            print(f"   Processed files: {result.get('processed_files', 0)}/{result.get('total_files', 0)}")
            if result.get('errors'):
                print(f"   Errors: {len(result['errors'])} files had issues")
            return True
        else:
            print(f"❌ Failed to create graph: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating graph: {str(e)}")
        return False

# Global variable for coding agent flow
coding_agent_flow = None

def process_query_and_update_summary(user_query: str, working_dir: str, args, context_manager: ContextManager, past_summary: str = None):
    """
    Process a query and update the conversation summary
    """
    global coding_agent_flow
    
    # Initialize shared dictionary
    shared = {
        "user_query": user_query,
        "working_dir": working_dir,
        "history": [],
        "graphrag": args.graphrag,
        "response": None,
        "past_summary": past_summary
    }
    
    try:
        # Run the flow
        coding_agent_flow.run(shared)
        
        # Get the response
        response = shared.get("response", "No response generated")
        
        # Update the conversation summary
        new_summary = context_manager.summarize_conversation(user_query, response, past_summary)
        
        # Reset history with the summary
        history = [{"Action": "previous conversation summary", "tool": "finish", "result": new_summary}]
        
        return new_summary, history
        
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        return past_summary, []

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
    parser.add_argument('--create-graph', '-c', type=str, 
                        help='Path to file or folder to add to the graph')
    args = parser.parse_args()
    
    # Ensure working directory exists
    working_dir = os.path.abspath(args.working_dir)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir, exist_ok=True)
        print(f"📁 Created working directory: {working_dir}")
    
    # Initialize ContextManager
    context_manager = ContextManager()
    
    # Initialize global coding agent flow
    global coding_agent_flow
    coding_agent_flow = create_main_flow(args.graphrag, args.model)
    
    # Handle create-graph argument
    if args.create_graph:
        if create_graph_from_path(args.create_graph, working_dir, args.model):
            print("✅ Graph creation completed successfully!")
        else:
            print("❌ Graph creation failed!")
        return
    
    # If query provided via command line, process it
    if args.query:
        user_query = args.query
        past_summary = None
        
        new_summary, history = process_query_and_update_summary(user_query, working_dir, args, context_manager, past_summary)
        return
    
    # Interactive mode - ask for queries
    print(f"🤖 GraphRAG Coding Agent")
    print(f"📁 Working Directory: {working_dir}")
    print(f"🧠 Model: {args.model}")
    print("\n💡 Usage:")
    print("  - Type '/graph <path>' to add files/folders to the knowledge graph")
    print("  - Type any other query to ask questions about your codebase")
    print("  - Type 'exit' to quit")
    print("-" * 50)
    
    # Initialize conversation tracking
    past_summary = None
    
    while True:
        user_query = input("\nWhat would you like me to help you with? ").strip()
        
        if user_query.lower() == 'exit':
            print("👋 Goodbye!")
            break
        elif not user_query:
            continue
        
        # Check if this is a graph creation command
        if user_query.startswith('/graph'):
            # Extract the path from the command
            path_part = user_query[7:].strip()  # Remove '/graph ' prefix
            
            if not path_part:
                print("❌ Please provide a path after /graph")
                print("Example: /graph ./src")
                continue
            
            # Handle relative paths
            if not os.path.isabs(path_part):
                path_part = os.path.join(working_dir, path_part)
            
            create_graph_from_path(path_part, working_dir, args.model)
            
        else:
            # Regular query - process and update summary
            new_summary, history = process_query_and_update_summary(user_query, working_dir, args, context_manager, past_summary)
            past_summary = new_summary

if __name__ == "__main__":
    main()