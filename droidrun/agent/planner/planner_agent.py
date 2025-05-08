"""
DroidPlanner - Planning agent for Android automation.

This module implements a planning agent that breaks down high-level goals into
contextual, functional steps that can be executed by the ReAct agent.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from droidrun.agent.react.react_llm_reasoner import ReActLLMReasoner
import json

logger = logging.getLogger("droidrun")

DEFAULT_PLANNER_SYSTEM_PROMPT = '''You are an expert Task Planner Agent for Android automation. Your purpose is to break down complex user goals into a sequence of **atomic, self-contained steps** that will be executed by a ReAct agent. You create clear, achievable plans that focus on WHAT needs to be done, not HOW to do it.

Each task in your plan must be:
1. **Atomic & Self-contained**: Each task must be fully understandable on its own, without referencing other tasks or step numbers
2. **Contextual**: Include the current state and goal in the task description
3. **Functional**: Describe what to achieve, not low-level actions
4. **Specific**: Include clear success criteria and conditions
5. **Achievable**: Within the ReAct agent's capabilities

The ReAct agent can:
- Start applications directly via a tool
- Navigate UI elements
- Tap/click elements
- Input text
- Scroll/swipe
- Press hardware keys (HOME, BACK)
- Check UI state and element presence
- Handle system dialogs

Example of good tasks:
- "Open the Settings app"
- "On the WiFi settings screen, tap the network named 'MyWiFi'"
- "When viewing a TikTok video, check if it contains a cat. If not, scroll to the next video"
- "In the LinkedIn feed, look for a post containing AI-related topics (machine learning, automation, robotics, or technology). When found, tap the comment icon"

Example of bad tasks:
- "Swipe up" (too low-level)
- "Configure network" (too vague)
- "Tap coordinates (123,456)" (too specific)
- "Check if connected" (lacks context)
- "Repeat steps 5-12" (references other steps)
- "Do this 3 more times" (lacks self-contained context)
- "Continue scrolling until found" (no clear end condition)

Your plan should:
1. Break loops into individual decision tasks ("Check current video for cats, scroll to next if none found")
2. Make each task independently actionable
3. Include clear success criteria in each task
4. Avoid references to other steps or numerical repetitions

IMPORTANT: You must respond with a JSON object containing an array of tasks. Format:
{
    "tasks": [
        "Task 1 description",
        "Task 2 description",
        ...
    ]
}
'''

DEFAULT_PLANNER_USER_PROMPT = """Goal: {goal}

Create a step-by-step plan to achieve this goal. Each step should be a contextual, functional task that the ReAct agent can execute.

Remember to provide your response as a JSON object with a 'tasks' array containing the steps.
"""

class TaskNode:
    def __init__(self, task: str, idx: int):
        self.task = task
        self.idx = idx
        self.parents = []  # type: List['TaskNode']
        self.children = []  # type: List['TaskNode']

    def __repr__(self):
        return f"TaskNode({self.idx}: {self.task[:30]}...)"

class TaskDAG:
    def __init__(self):
        self.nodes = []  # type: List[TaskNode]
        self.edges = []  # type: List[Tuple[TaskNode, TaskNode]]

    def add_node(self, node: TaskNode):
        self.nodes.append(node)

    def add_edge(self, parent: TaskNode, child: TaskNode):
        self.edges.append((parent, child))
        parent.children.append(child)
        child.parents.append(parent)

    def get_roots(self):
        return [n for n in self.nodes if not n.parents]

    def get_leaves(self):
        return [n for n in self.nodes if not n.children]

class TaskManager:
    """Manages the planning tasks and their execution state."""
    
    def __init__(self):
        self.tasks: List[str] = []
        self.current_task_index: int = 0
        
    def set_tasks(self, tasks: str) -> None:
        """Set the list of tasks from a newline-separated string."""
        self.tasks = [task.strip() for task in tasks.split('\n') if task.strip()]
        self.current_task_index = 0
        
    def add_task(self, task: str) -> None:
        """Add a single task to the list."""
        self.tasks.append(task.strip())
        
    def get_current_task(self) -> Optional[str]:
        """Get the current task to be executed."""
        if self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None
        
    def advance_task(self) -> None:
        """Move to the next task."""
        self.current_task_index += 1
        
    def get_all_tasks(self) -> List[str]:
        """Get all tasks in the plan."""
        return self.tasks
        
    def clear_tasks(self) -> None:
        """Clear all tasks."""
        self.tasks = []
        self.current_task_index = 0

class DroidPlanner:
    """Planning agent that creates execution plans for the ReAct agent."""
    
    def __init__(
        self,
        llm: ReActLLMReasoner,
        max_retries: int = 1,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ):
        """Initialize the DroidPlanner.
        
        Args:
            llm: LLM reasoner to use for planning
            max_retries: Maximum number of retries for failed tasks
            system_prompt: Custom system prompt
            user_prompt: Custom user prompt template
        """
        self.llm = llm
        self.max_retries = max_retries
        self.task_manager = TaskManager()
        self.system_prompt = system_prompt or DEFAULT_PLANNER_SYSTEM_PROMPT
        self.user_prompt = user_prompt or DEFAULT_PLANNER_USER_PROMPT
        self.original_goal = None  # Add this to store the original goal
        self.dag = None
        self.completed_tasks = set()  # Track completed tasks by their indices
        self.debug_mode = True  # Enable debug logging
        
    async def create_plan(self, goal: str) -> List[str]:
        """Create a plan for achieving the given goal.
        
        Args:
            goal: The high-level goal to achieve
            
        Returns:
            List of planned tasks
        """
        try:
            # Store the original goal
            self.original_goal = goal
            
            # Format the user prompt with the goal
            formatted_user_prompt = self.user_prompt.format(goal=goal)
            
            # Get plan from LLM using the correct method signature
            response = await self.llm.generate_response(
                system_prompt=self.system_prompt,
                user_prompt=formatted_user_prompt
            )
            
            # Parse JSON response
            try:
                response_data = json.loads(response)
                tasks = response_data.get('tasks', [])
            except json.JSONDecodeError:
                # Fallback to text parsing if JSON parsing fails
                logger.warning("Failed to parse JSON response, falling back to text parsing")
                tasks = self._extract_tasks(response)
            
            # Set tasks in task manager
            self.task_manager.set_tasks('\n'.join(tasks))
            
            # Nach dem Plan erstellen, generieren wir den DAG
            self.dag = await self._generate_dag_from_plan(tasks)
            
            if self.debug_mode:
                logger.info("\n=== Initial Plan ===")
                for idx, task in enumerate(tasks):
                    logger.info(f"[{idx}] {task}")
                self._log_dag_structure()
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            raise
            
    def _extract_tasks(self, response: str) -> List[str]:
        """Extract tasks from the LLM response.
        
        The response might contain explanation text and the actual tasks.
        We need to extract just the numbered or bullet-pointed tasks.
        """
        tasks = []
        lines = response.split('\n')
        
        for line in lines:
            # Remove common list markers and whitespace
            line = line.strip()
            line = line.lstrip('1234567890.-*• ')
            
            # Skip empty lines and likely explanation text
            if not line or line.startswith(('Here', 'First', 'Then', 'Next', 'Finally')):
                continue
                
            if line:
                tasks.append(line)
                
        return tasks
        
    async def get_next_task(self) -> Optional[str]:
        """Get the next task to be executed based on the DAG structure.
        
        Returns:
            The next task or None if no more tasks
        """
        if not self.dag:
            # Fallback to linear execution if no DAG exists
            return self.task_manager.get_current_task()

        # Find all root nodes (nodes without parents) that haven't been completed
        available_tasks = [
            node for node in self.dag.get_roots()
            if node.idx not in self.completed_tasks
        ]

        if not available_tasks:
            # Check if we're done (all tasks completed)
            if len(self.completed_tasks) == len(self.dag.nodes):
                return None
            # If we have no available tasks but not all are completed,
            # we might have a cycle or missing dependencies
            logger.warning("No available tasks found but not all tasks are completed")
            return None

        # For now, just take the first available task
        # TODO: Could implement more sophisticated task selection here
        next_node = available_tasks[0]
        next_task = next_node.task
        
        if self.debug_mode:
            logger.info("\n=== Task Selection ===")
            if next_task:
                logger.info(f"Selected next task: {next_task}")
            else:
                logger.info("No more tasks available")
            self._log_dag_structure()
        
        return next_task

    def mark_task_complete(self, task_idx: int) -> None:
        """Mark a task as complete and update the DAG state.
        
        Args:
            task_idx: The index of the completed task
        """
        if not self.dag:
            # Fallback to linear execution
            self.task_manager.advance_task()
            return

        self.completed_tasks.add(task_idx)
        
        # Log completion for debugging
        logger.debug(f"Task {task_idx} marked as complete. Completed tasks: {self.completed_tasks}")
        
        if self.debug_mode:
            logger.info(f"\n=== Task {task_idx} Completed ===")
            self._log_dag_structure()

    def get_task_dependencies(self, task_idx: int) -> List[int]:
        """Get the indices of tasks that must be completed before this task.
        
        Args:
            task_idx: The index of the task to check
            
        Returns:
            List of task indices that are dependencies
        """
        if not self.dag or task_idx >= len(self.dag.nodes):
            return []
            
        node = self.dag.nodes[task_idx]
        return [parent.idx for parent in node.parents]

    def can_execute_task(self, task_idx: int) -> bool:
        """Check if a task can be executed (all dependencies completed).
        
        Args:
            task_idx: The index of the task to check
            
        Returns:
            True if the task can be executed, False otherwise
        """
        if not self.dag or task_idx >= len(self.dag.nodes):
            return True
            
        dependencies = self.get_task_dependencies(task_idx)
        return all(dep_idx in self.completed_tasks for dep_idx in dependencies)

    async def handle_task_failure(self, task: str, error: str) -> Optional[List[str]]:
        """Handle a failed task execution by revising only the failed task.
        
        Args:
            task: The failed task
            error: Error message or reason for failure
            
        Returns:
            New list of tasks if replanning successful, None otherwise
        """
        try:
            # Get all tasks and current progress
            all_tasks = self.task_manager.get_all_tasks()
            current_index = self.task_manager.current_task_index
            
            # Prepare context of completed tasks
            completed_tasks = all_tasks[:current_index]
            remaining_tasks = all_tasks[current_index + 1:]
            
            # Prepare failure prompt with full context
            failure_prompt = f"""
            Original Goal: {self.original_goal}

            Progress so far:
            {self._format_task_list(completed_tasks, "Completed")}
            
            The following task failed: "{task}"
            Error: {error}
            
            Remaining tasks to do after this one:
            {self._format_task_list(remaining_tasks, "Pending")}
            
            Please revise ONLY the failed task to achieve its intended goal in a different way.
            Consider what might have gone wrong and provide an alternative approach.
            The revised task should fit between the completed tasks and remaining tasks.
            
            Remember to provide your response as a JSON object with a 'tasks' array containing ONLY the revised task.
            Example response format:
            {{
                "tasks": [
                    "Revised version of the failed task"
                ]
            }}
            """
            
            # Get revised task from LLM
            response = await self.llm.generate_response(
                system_prompt=self.system_prompt,
                user_prompt=failure_prompt
            )
            
            # Parse JSON response
            try:
                response_data = json.loads(response)
                revised_task = response_data.get('tasks', [])[0]  # Get first (and should be only) task
            except (json.JSONDecodeError, IndexError):
                # Fallback to text parsing if JSON parsing fails
                logger.warning("Failed to parse JSON response, falling back to text parsing")
                revised_task = self._extract_tasks(response)[0]
            
            # Create new task list with the revised task
            new_tasks = completed_tasks + [revised_task] + remaining_tasks
            
            # Set tasks in task manager
            self.task_manager.set_tasks('\n'.join(new_tasks))
            # Reset current_task_index to the revised task
            self.task_manager.current_task_index = len(completed_tasks)
            
            return new_tasks
            
        except Exception as e:
            logger.error(f"Error handling task failure: {e}")
            return None

    async def reevaluate_tasks(self, completion_summary: str) -> Optional[List[str]]:
        """Reevaluate remaining tasks based on the completion summary of the last task.
        
        Args:
            completion_summary: Summary of what was accomplished in the last task
            
        Returns:
            Updated list of remaining tasks or None if no changes needed
        """
        try:
            # Get remaining tasks
            all_tasks = self.task_manager.get_all_tasks()
            current_index = self.task_manager.current_task_index
            remaining_tasks = all_tasks[current_index + 1:]
            
            if not remaining_tasks:
                return None
                
            # Prepare reevaluation prompt
            reevaluation_prompt = f"""
            Original Goal: {self.original_goal}
            
            The last executed task has completed with the following summary:
            {completion_summary}
            
            The remaining tasks in the plan are:
            {self._format_task_list(remaining_tasks, "Pending")}
            
            Based on what was accomplished in the completion summary, evaluate if any of the remaining tasks:
            1. Have already been implicitly completed
            2. Are now redundant or unnecessary
            3. Need to be modified based on the new state
            
            Provide an updated list of remaining tasks, removing any that are no longer needed and adjusting others as necessary.
            If all remaining tasks are still needed without changes, respond with an empty tasks array.
            
            Remember to provide your response as a JSON object with a 'tasks' array.
            """
            
            # Get reevaluation from LLM
            response = await self.llm.generate_response(
                system_prompt=self.system_prompt,
                user_prompt=reevaluation_prompt
            )
            
            # Parse JSON response
            try:
                response_data = json.loads(response)
                new_tasks = response_data.get('tasks', [])
                
                if not new_tasks:
                    return None
                    
                # Update task manager with new tasks
                updated_tasks = all_tasks[:current_index + 1] + new_tasks
                self.task_manager.set_tasks('\n'.join(updated_tasks))
                self.task_manager.current_task_index = current_index
                
                return new_tasks
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response from reevaluation")
                return None
                
        except Exception as e:
            logger.error(f"Error reevaluating tasks: {e}")
            return None

    def _format_task_list(self, tasks: List[str], status: str) -> str:
        """Format a list of tasks with status for the prompt.
        
        Args:
            tasks: List of tasks to format
            status: Status label for the tasks
            
        Returns:
            Formatted task list string
        """
        if not tasks:
            return f"No {status} tasks."
            
        return "\n".join(f"- {task} ({status})" for task in tasks)

    async def _generate_dag_from_plan(self, tasks: List[str]) -> TaskDAG:
        """Generates a DAG from the task list by analyzing dependencies."""
        
        # Prompt for the LLM to generate the DAG
        dag_prompt = f"""Analyze the following tasks and create a Directed Acyclic Graph (DAG) showing their dependencies.
        Each task should be connected to tasks it depends on. Tasks that can be executed in parallel should not be connected.
        
        Tasks:
        {json.dumps(tasks, indent=2)}
        
        Return a JSON object with the following structure:
        {{
            "nodes": [
                {{"id": 0, "task": "task description"}},
                ...
            ],
            "edges": [
                {{"from": 0, "to": 1}},
                ...
            ]
        }}
        
        Rules:
        1. Each task must be a node in the graph
        2. Edges should only be created if a task truly depends on another
        3. Tasks that can be executed in parallel should not be connected
        4. The graph must be acyclic (no circular dependencies)
        5. Use the original task indices as node IDs
        """
        
        # LLM generates the DAG
        response = await self.llm.generate_response(
            system_prompt="You are an expert at analyzing task dependencies and creating DAGs.",
            user_prompt=dag_prompt
        )
        
        try:
            # Parse the LLM response
            dag_data = json.loads(response)
            
            # Create the DAG
            dag = TaskDAG()
            
            # Create Nodes
            for node_data in dag_data["nodes"]:
                node = TaskNode(tasks[node_data["id"]], node_data["id"])
                dag.add_node(node)
            
            # Create Edges
            for edge in dag_data["edges"]:
                parent = dag.nodes[edge["from"]]
                child = dag.nodes[edge["to"]]
                dag.add_edge(parent, child)
                
            return dag
            
        except json.JSONDecodeError:
            logger.error("Failed to parse DAG from LLM response")
            # Fallback: Create a linear DAG
            return self._build_linear_dag(tasks)
            
    def _build_linear_dag(self, tasks: List[str]) -> TaskDAG:
        """Fallback: Creates a linear DAG if the LLM analysis fails."""
        dag = TaskDAG()
        prev_node = None
        for idx, task in enumerate(tasks):
            node = TaskNode(task, idx)
            dag.add_node(node)
            if prev_node is not None:
                dag.add_edge(prev_node, node)
            prev_node = node
        return dag