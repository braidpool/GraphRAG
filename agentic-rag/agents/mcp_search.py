# server.py
from fastmcp import FastMCP, Context
import requests


mcp = FastMCP("Code Research Server")


@mcp.tool(name="search_stackoverflow")
def search_stackoverflow(query: str, limit: int = 5) -> list[dict]:
    """Search StackOverflow for questions and their top answers.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
        
    Returns:
        List of dicts containing question, answer, and metadata
    """
    try:
        print(f"Searching StackOverflow for: {query}")
        
        # 1. Search questions
        search_params = {
            "order": "desc",
            "sort": "relevance",
            "site": "stackoverflow",
            "q": query,
            "pagesize": limit,
            "filter": "withbody"
        }
        
        print(f"Sending request to StackOverflow API with params: {search_params}")
        resp = requests.get(
            "https://api.stackexchange.com/2.3/search/advanced",
            params=search_params,
            timeout=30
        )
        print(f"StackOverflow API response status: {resp.status_code}")
        print(f"Response headers: {resp.headers}")
        
        resp.raise_for_status()
        data = resp.json()
        
        if "items" not in data:
            print(f"Unexpected StackOverflow API response: {data}")
            return []
            
        questions = data["items"]
        print(f"Found {len(questions)} questions")
        
        if not questions:
            print("No questions found in StackOverflow response")
            return []

        # 2. Fetch answers with body
        ids = ";".join(str(q["question_id"]) for q in questions)
        answer_params = {
            "site": "stackoverflow",
            "filter": "withbody",
            "order": "desc",
            "sort": "votes"
        }
        
        print(f"Fetching answers for question IDs: {ids}")
        ans_resp = requests.get(
            f"https://api.stackexchange.com/2.3/questions/{ids}/answers",
            params=answer_params,
            timeout=30
        )
        print(f"Answers API response status: {ans_resp.status_code}")
        
        ans_resp.raise_for_status()
        answers_data = ans_resp.json()
        answers = answers_data.get("items", [])
        print(f"Found {len(answers)} answers")

        # 3. Process and combine results
        ans_by_q = {}
        for a in answers:
            ans_by_q.setdefault(a["question_id"], []).append(a)

        def pick_best(ans_list):
            if not ans_list:
                return None
            # First try accepted answer, then highest voted
            accepted = next((a for a in ans_list if a.get("is_accepted")), None)
            if accepted:
                return accepted
            return max(ans_list, key=lambda x: x.get("score", 0), default=None)

        results = []
        for q in questions:
            qid = q["question_id"]
            ans = pick_best(ans_by_q.get(qid, []))
            if not ans:
                print(f"No suitable answer found for question ID: {qid}")
                continue
                
            result = {
                "platform": "stackoverflow",
                "title": q["title"],
                "url": q["link"],
                "content": ans.get("body", ""),
                "score": ans.get("score", 0),
                "is_accepted": ans.get("is_accepted", False),
                "tags": q.get("tags", []),
                "creation_date": q.get("creation_date"),
                "answer_count": q.get("answer_count", 0),
                "view_count": q.get("view_count", 0)
            }
            results.append(result)
            
        print(f"Returning {len(results)} results from StackOverflow")
        return results[:limit]
        
    except requests.RequestException as e:
        print(f"Request error in StackOverflow search: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return []
    except Exception as e:
        print(f"Unexpected error in StackOverflow search: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


@mcp.tool
def search_github(query: str, language: str = None, limit: int = 5) -> list[dict]:
    """Search GitHub for repositories matching the query.
    
    Args:
        query: Search query string
        language: Optional programming language filter
        limit: Maximum number of results to return (max 30)
        
    Returns:
        List of dicts containing repository information
    """
    try:
        print(f"Searching GitHub for: {query}")
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            # For higher rate limits, uncomment and add your GitHub token
            # "Authorization": f"token YOUR_GITHUB_TOKEN"
        }
        
        # Build search query with optional language filter
        search_query = query
        if language:
            search_query += f" language:{language}"
        
        # Limit to a reasonable number of results
        per_page = min(max(1, limit), 30)  # GitHub's max is 100 per page, but we'll use 30
        
        params = {
            "q": search_query,
            "per_page": per_page,
            "sort": "stars",
            "order": "desc"
        }
        
        print(f"Sending request to GitHub API with params: {params}")
        
        # Search repositories
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params=params,
            headers=headers,
            timeout=30
        )
        
        print(f"GitHub API response status: {resp.status_code}")
        print(f"Rate limit remaining: {resp.headers.get('x-ratelimit-remaining', 'unknown')}")
        
        resp.raise_for_status()
        
        data = resp.json()
        if "items" not in data:
            print(f"Unexpected GitHub API response: {data}")
            return []
            
        repos = data["items"]
        print(f"Found {len(repos)} repositories")
        
        results = []
        for repo in repos[:limit]:  # Ensure we respect the limit
            try:
                # Get additional repository details
                repo_details = {
                    "platform": "github",
                    "name": repo.get("full_name", ""),
                    "url": repo.get("html_url", ""),
                    "description": repo.get("description", ""),
                    "stars": repo.get("stargazers_count", 0),
                    "forks": repo.get("forks_count", 0),
                    "language": repo.get("language"),
                    "topics": repo.get("topics", []),
                    "created_at": repo.get("created_at"),
                    "updated_at": repo.get("pushed_at"),
                    "open_issues": repo.get("open_issues_count", 0),
                    "license": repo.get("license", {}).get("name") if repo.get("license") else None,
                }
                
                # Safely get owner info
                owner = repo.get("owner", {})
                if owner:
                    repo_details["owner"] = {
                        "login": owner.get("login", ""),
                        "url": owner.get("html_url", ""),
                        "type": owner.get("type", "")
                    }
                
                results.append(repo_details)
                
            except Exception as repo_error:
                print(f"Error processing repository: {str(repo_error)}")
                continue
            
        print(f"Returning {len(results)} results from GitHub")
        return results
        
    except requests.RequestException as e:
        error_msg = f"Request error in GitHub search: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f"\nStatus code: {e.response.status_code}"
            try:
                error_msg += f"\nResponse: {e.response.text}"
            except Exception:
                pass
        print(error_msg)
        return []
        
    except Exception as e:
        print(f"Unexpected error in GitHub search: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


@mcp.tool
def search_all(query: str, limit: int = 3, include_github: bool = True, 
               include_stackoverflow: bool = True) -> dict:
    """
    Search multiple platforms simultaneously and return combined results.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return per platform
        include_github: Whether to include GitHub results
        include_stackoverflow: Whether to include StackOverflow results
        
    Returns:
        Dict containing platform names as keys and search results as values
    """
    import concurrent.futures
    from typing import Dict, Any
    
    results: Dict[str, Any] = {}
    
    def safe_search(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        
        if include_github:
            futures['github'] = executor.submit(
                safe_search, search_github, query, None, limit
            )
            
        if include_stackoverflow:
            futures['stackoverflow'] = executor.submit(
                safe_search, search_stackoverflow, query, limit
            )
        
        # Wait for all searches to complete
        for platform, future in futures.items():
            try:
                results[platform] = future.result()
            except Exception as e:
                print(f"Error retrieving {platform} results: {e}")
                results[platform] = []
    
    # Combine and sort results by relevance (simple heuristic)
    combined = []
    for platform, items in results.items():
        for item in items:
            # Add a score for sorting
            if platform == 'github':
                # Higher stars = better
                item['_score'] = item.get('stars', 0) * 2
            else:  # stackoverflow
                # Higher score = better, with bonus for accepted answers
                item['_score'] = item.get('score', 0) * (2 if item.get('is_accepted') else 1)
            combined.append(item)
    
    # Sort by score in descending order
    combined.sort(key=lambda x: x.get('_score', 0), reverse=True)
    
    # Remove the temporary score field
    for item in combined:
        item.pop('_score', None)
    
    # Add platform counts to results
    platform_counts = {k: len(v) for k, v in results.items()}
    
    return {
        "results": combined[:limit * 2],  # Return more combined results
        "platform_counts": platform_counts,
        "total_results": len(combined),
        "query": query
    }

if __name__ == "__main__":
    
    mcp.run()
