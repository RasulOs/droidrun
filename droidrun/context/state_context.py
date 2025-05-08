"""
State Context Management System for DroidRun.

This module provides a flexible and type-safe way to manage different types of state
contexts that can be injected into agents during execution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TypeVar, Generic
from enum import Enum
from datetime import datetime

# Type variable for generic state values
T = TypeVar('T')

class AgentType(Enum):
    """Types of agents in the system."""
    REACT = "react"
    PLANNER = "planner"
    TASK_MANAGER = "task_manager"
    APP_STARTER = "app_starter"

class ContextType(Enum):
    """Types of contexts that can be managed."""
    AGENT = "agent"
    PLAN = "plan"
    DEVICE = "device"
    MEMORY = "memory"
    GLOBAL = "global"

@dataclass
class BaseContext:
    """Base class for all contexts."""
    context_id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeviceState:
    """Represents the current state of the Android device."""
    ui_state: Dict[str, Any]
    phone_state: Dict[str, Any]
    screenshot_data: Optional[bytes] = None
    installed_packages: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class AgentState:
    """Represents the current state of an agent."""
    agent_type: AgentType
    current_goal: Optional[str] = None
    step_count: int = 0
    action_count: int = 0
    last_action: Optional[str] = None
    last_result: Optional[Any] = None
    is_active: bool = False

@dataclass
class TaskState:
    """Represents the state of a task."""
    task_id: str
    description: str
    status: str  # "pending", "in_progress", "completed", "failed"
    expected_outcome: str
    actual_outcome: Optional[str] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PlanState:
    """Represents the current state of an execution plan."""
    plan_id: str
    original_goal: str
    tasks: List[TaskState]
    current_task_index: int = 0
    is_complete: bool = False
    has_failed: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class MemoryState:
    """Represents the current state of agent memories."""
    memories: List[str] = field(default_factory=list)
    context_window: List[Dict[str, Any]] = field(default_factory=list)
    max_memories: int = 100
    max_context_window: int = 10

@dataclass
class StateContext(Generic[T]):
    """Generic container for state context."""
    context_type: ContextType
    state: T
    metadata: Dict[str, Any] = field(default_factory=dict)

class StateContextManager:
    """Manages different types of state contexts."""
    
    def __init__(self):
        """Initialize the state context manager."""
        self._contexts: Dict[str, StateContext] = {}
        self._device_state = DeviceState(ui_state={}, phone_state={})
        self._agent_states: Dict[str, AgentState] = {}
        self._plan_states: Dict[str, PlanState] = {}
        self._memory_state = MemoryState()
        
    def create_agent_context(
        self,
        agent_id: str,
        agent_type: AgentType,
        goal: Optional[str] = None
    ) -> StateContext[AgentState]:
        """Create a new agent context.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of the agent
            goal: Optional goal for the agent
            
        Returns:
            New agent context
        """
        state = AgentState(
            agent_type=agent_type,
            current_goal=goal,
            is_active=True
        )
        context = StateContext(
            context_type=ContextType.AGENT,
            state=state
        )
        self._contexts[f"agent:{agent_id}"] = context
        return context
        
    def create_plan_context(
        self,
        plan_id: str,
        goal: str,
        tasks: List[TaskState]
    ) -> StateContext[PlanState]:
        """Create a new plan context.
        
        Args:
            plan_id: Unique identifier for the plan
            goal: The plan's goal
            tasks: List of tasks in the plan
            
        Returns:
            New plan context
        """
        state = PlanState(
            plan_id=plan_id,
            original_goal=goal,
            tasks=tasks
        )
        context = StateContext(
            context_type=ContextType.PLAN,
            state=state
        )
        self._contexts[f"plan:{plan_id}"] = context
        return context
        
    def update_device_state(
        self,
        ui_state: Optional[Dict[str, Any]] = None,
        phone_state: Optional[Dict[str, Any]] = None,
        screenshot_data: Optional[bytes] = None,
        installed_packages: Optional[List[Dict[str, str]]] = None
    ) -> StateContext[DeviceState]:
        """Update the device state context.
        
        Args:
            ui_state: Current UI state
            phone_state: Current phone state
            screenshot_data: Optional screenshot data
            installed_packages: List of installed packages
            
        Returns:
            Updated device context
        """
        if ui_state is not None:
            self._device_state.ui_state = ui_state
        if phone_state is not None:
            self._device_state.phone_state = phone_state
        if screenshot_data is not None:
            self._device_state.screenshot_data = screenshot_data
        if installed_packages is not None:
            self._device_state.installed_packages = installed_packages
            
        context = StateContext(
            context_type=ContextType.DEVICE,
            state=self._device_state
        )
        self._contexts["device"] = context
        return context
        
    def add_memory(self, memory: str) -> None:
        """Add a new memory to the memory state.
        
        Args:
            memory: Memory to add
        """
        self._memory_state.memories.append(memory)
        if len(self._memory_state.memories) > self._memory_state.max_memories:
            self._memory_state.memories.pop(0)
            
    def update_context_window(self, context: Dict[str, Any]) -> None:
        """Update the context window in memory state.
        
        Args:
            context: Context to add to the window
        """
        self._memory_state.context_window.append(context)
        if len(self._memory_state.context_window) > self._memory_state.max_context_window:
            self._memory_state.context_window.pop(0)
            
    def get_context(self, context_id: str) -> Optional[StateContext]:
        """Get a context by its ID.
        
        Args:
            context_id: ID of the context to retrieve
            
        Returns:
            The context if found, None otherwise
        """
        return self._contexts.get(context_id)
        
    def get_agent_context(self, agent_id: str) -> Optional[StateContext[AgentState]]:
        """Get an agent's context.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            The agent's context if found, None otherwise
        """
        return self._contexts.get(f"agent:{agent_id}")
        
    def get_plan_context(self, plan_id: str) -> Optional[StateContext[PlanState]]:
        """Get a plan's context.
        
        Args:
            plan_id: ID of the plan
            
        Returns:
            The plan's context if found, None otherwise
        """
        return self._contexts.get(f"plan:{plan_id}")
        
    def get_device_context(self) -> StateContext[DeviceState]:
        """Get the current device context.
        
        Returns:
            The current device context
        """
        return StateContext(
            context_type=ContextType.DEVICE,
            state=self._device_state
        )
        
    def get_memory_context(self) -> StateContext[MemoryState]:
        """Get the current memory context.
        
        Returns:
            The current memory context
        """
        return StateContext(
            context_type=ContextType.MEMORY,
            state=self._memory_state
        )
        
    def update_task_state(
        self,
        plan_id: str,
        task_id: str,
        status: Optional[str] = None,
        actual_outcome: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """Update a task's state in a plan.
        
        Args:
            plan_id: ID of the plan containing the task
            task_id: ID of the task to update
            status: New status for the task
            actual_outcome: Actual outcome of the task
            error: Error message if the task failed
        """
        if plan_context := self.get_plan_context(plan_id):
            plan_state = plan_context.state
            for task in plan_state.tasks:
                if task.task_id == task_id:
                    if status:
                        task.status = status
                    if actual_outcome:
                        task.actual_outcome = actual_outcome
                    if error:
                        task.error = error
                    plan_state.updated_at = datetime.now()
                    break
                    
    def clear_context(self, context_id: str) -> None:
        """Remove a context from the manager.
        
        Args:
            context_id: ID of the context to remove
        """
        self._contexts.pop(context_id, None)
        
    def clear_all_contexts(self) -> None:
        """Clear all contexts from the manager."""
        self._contexts.clear()
        self._device_state = DeviceState(ui_state={}, phone_state={})
        self._memory_state = MemoryState() 