---
title: "OpenCode: Why It's Different and How to Use It with Python"
description: Learn how OpenCode works as programmable AI coding infrastructure. Built in TypeScript, controlled with Python — covering agents, skills, PLAN.md, and MLflow integration.
author: kareem
date: 2026-02-19
draft: false
featured: false
categories:
  - blogging
  - til
  - opencode
  - ai
  - dspy
  - blog/build/project
image: images/Opencode.jpg
---

## OpenCode: Why It's Different and How to Use It with Python

OpenCode is an AI coding assistant. But unlike Cursor or Claude Code, it's built to be extended and controlled programmatically.

Here's what makes it different and why that matters.

---

## What is OpenCode?

OpenCode is an AI coding assistant. Different from Cursor, Claude Code, and Copilot.

**Key architecture features:**

**Works everywhere**

- Terminal
- IDE extensions
- Desktop app

**LSP-enabled**
Automatically loads Language Server Protocol for your project. LLM gets proper code intelligence.

**Multi-session**
Run multiple agents in parallel on the same project. No conflicts.

**Share links**
Every session gets a shareable link. Good for debugging and code review.

**Model flexibility**

- GitHub Copilot account
- ChatGPT Plus/Pro account
- 75+ LLM providers via Models.dev
- Local models

**The key difference**: OpenCode is infrastructure you can build on, not just a chat interface.

---

## OpenCode is Programmer Friendly

OpenCode is built in TypeScript. But you can control it with Python. No TypeScript needed.

**How I use it:**

1. Run OpenCode as background process on specific port
2. Create Python client that controls the server
3. Use it with DSPy for smarter workflows

**Example setup:**

```bash
opencode server --port 3000
```

```python
# Python client
class OpenCodeClient:
    def __init__(self, port=3000):
        self.base_url = f"http://localhost:{port}"
    
    def invoke_agent(self, agent, prompt, context=None):
        response = httpx.post(
            f"{self.base_url}/api/sessions",
            json={
                "agent": agent,
                "prompt": prompt,
                "context": context
            }
        )
        return response.json()
```

Now you have programmatic control. Build whatever workflows you need.

---

## Why This Hybrid Approach?

You could try to simulate OpenCode features in DSPy tool calling. But it won't be as mature or robust.

**What you'd need to rebuild:**

- LSP integration
- Multi-session management
- File operations with error handling
- Bash execution with timeouts

OpenCode already has this. Battle-tested.

**What DSPy adds:**
Structured programming and optimization for AI workflows.

**Example workflow:**

```python
class FigmaToCodeWorkflow(dspy.Module):
    def forward(self, figma_url):
        # DSPy: Analyze Figma design
        analysis = self.analyzer(figma_url=figma_url)
        
        # DSPy: Pick best component
        selection = self.selector(design_analysis=analysis)
        
        # OpenCode: Build it
        result = opencode_client.invoke_agent(
            agent="build",
            prompt=f"Create component: {selection.component_name}",
            context=selection.design_specs
        )
        
        return result
```

DSPy analyzes and decides. OpenCode executes with proper LSP support.

---

## OpenCode Tools

OpenCode has built-in tools:

- `read` - Read files
- `write` - Create/overwrite files
- `edit` - Modify existing files
- `bash` - Execute shell commands
- `grep` - Search file contents
- `glob` - Find files by pattern
- `lsp` - Code intelligence (definitions, references)
- `webfetch` - Fetch web content

**You can create custom tools.**

Example custom tool:

```json
{
  "tools": {
    "analyze_performance": {
      "command": "python scripts/analyze.py",
      "description": "Analyze code performance"
    }
  }
}
```

**You can also use MCP servers** for database access, API integrations, and third-party services.

---

## Specialized Agents

OpenCode has built-in agents:

**Build Agent**

- Full access to all tools
- Can edit files and run bash commands
- Default agent for development

**Plan Agent**

- Read-only by default
- All edits set to "ask" permission
- Good for analysis without changes

**General Agent**

- Multi-step tasks
- Full tool access
- Invoke with `@general`

**Explore Agent**

- Fast codebase exploration
- Read-only
- Find files and search code

**Create custom agents:**

```json
{
  "agent": {
    "security-auditor": {
      "description": "Reviews code for vulnerabilities",
      "mode": "subagent",
      "model": "anthropic/claude-sonnet-4",
      "permission": {
        "edit": "deny",
        "bash": "deny",
        "read": "allow"
      }
    }
  }
}
```

Each agent can have:

- Different models
- Different tools
- Different permissions
- Custom skills
- MCP servers

---

## OpenCode Rules (AGENTS.md)

Create `AGENTS.md` in your project root:

```markdown
# My Project

## Architecture
- `backend/` - FastAPI server
- `frontend/` - React app
- `agents/` - DSPy workflows

## Standards
- TypeScript strict mode
- All routes in `backend/routes/`
- Tests required

## Context
This is a collaborative coding platform.
```

Every agent reads this on startup. They understand your project structure and conventions.

**Types of rules:**

**Project rules**: `AGENTS.md` in project root

- Commits to Git
- Shared with team

**Global rules**: `~/.config/opencode/AGENTS.md`

- Personal preferences
- Not shared

**Advanced**: Reference external files or remote URLs:

```json
{
  "instructions": [
    "docs/standards.md",
    "https://raw.githubusercontent.com/org/rules/main/style.md"
  ]
}
```

---

## OpenCode Skills

Skills are reusable `SKILL.md` files. Agents can load them on demand.

Create skills in:

- Global: `~/.config/opencode/skills/`
- Project: `.opencode/skills/`

Example skill:

```markdown
<!-- .opencode/skills/react-component.md -->
# React Component Skill

Create React components following our patterns:
- Functional components with hooks
- TypeScript for props
- Tailwind for styling
```

Agent loads skill:

```
@build use react-component skill to create login form
```

Good for:

- Coding patterns
- Project conventions
- Reusable workflows

---

## Smart Planning with PLAN.md

Problem: Multiple agents working in parallel duplicate work or miss dependencies.

Solution: Shared `PLAN.md` file.

**How it works:**

Create `PLAN.md`:

```markdown
# Project Plan

## Authentication [IN PROGRESS - @build-agent-1]
- [x] JWT tokens
- [ ] Refresh logic (CURRENT)
- [ ] Password reset

## Frontend [BLOCKED - waiting on auth]
- [ ] Login form
- [ ] Dashboard
```

**Agent workflow:**

Before work:

- Read PLAN.md
- Find available tasks
- Mark [IN PROGRESS - @agent-name]

During work:

- Complete subtask
- Mark [x]
- Add new subtasks

After completion:

- Mark section complete
- Check if blocked tasks can proceed

**Impact:**

Before PLAN.md:

- 40% wasted work
- Constant manual coordination

After PLAN.md:

- <5% wasted work
- 90% self-coordination

**Real example from my Replit clone:**

I had agents building features out of order. Agent 1 builds auth while Agent 2 builds login UI (no endpoints exist yet).

Added PLAN.md with dependencies. Agents now check what's available and what's blocked. Problem solved.

---

## The Impact

My project timeline: 2 months with OpenCode vs 12 months without.

**Time saved:**

**Infrastructure (3 months)**

- LSP integration
- Multi-session management
- File operations
- Bash execution

OpenCode ships all of this.

**Coordination (6 months)**

- PLAN.md pattern for agent sync
- Multi-session for parallel work
- Discovering this pattern would take months

**Debugging (1 month)**

- MLflow caught issues early
- Prompt optimization
- Performance improvements

**What made the difference:**

- Not rebuilding mature infrastructure
- Programmatic control with Python
- Agent coordination patterns
- Observability from day one

---

## MLflow Integration

Attach MLflow to your Python client:

```python
import mlflow

class OpenCodeClient:
    def invoke_agent(self, agent, prompt, context=None):
        with mlflow.start_run():
            # Log inputs
            mlflow.log_param("agent", agent)
            mlflow.log_param("prompt_length", len(prompt))
            
            # Call OpenCode
            result = self._call_opencode(agent, prompt, context)
            
            # Log outputs
            mlflow.log_metric("execution_time", result.duration)
            mlflow.log_metric("files_modified", len(result.changes))
            mlflow.log_text(result.output, "output.txt")
            
            return result
```

**What you get:**

Real-time dashboard:

- Which agents are running
- What they're working on
- Operation duration
- Success/failure rates

Historical data:

- Slow agents
- Prompt optimization opportunities
- Debugging failures
- Performance trends

Cost tracking:

- Token usage per agent
- Model costs per feature
- ROI on different approaches

**My setup:**

Use decorators for clean code:

```python
@mlflow.trace(name="figma_to_code", span_type=SpanType.CHAIN)
async def figma_to_code(figma_url):
    # Automatically traced
    pass
```

Key tags I track:

- `project_name`, `figma_url`
- `model`, `provider`
- Session ID for debugging

---

## Getting Started

1. **Install OpenCode**

```bash
npm install -g opencode
```

1. **Start server**

```bash
opencode server --port 3000
```

1. **Create Python client**

```python
import httpx

class OpenCodeClient:
    def __init__(self, port=3000):
        self.base_url = f"http://localhost:{port}"
    
    def invoke_agent(self, agent, prompt, context=None):
        response = httpx.post(
            f"{self.base_url}/api/sessions",
            json={
                "agent": agent,
                "prompt": prompt,
                "context": context
            }
        )
        return response.json()
```

1. **Set up AGENTS.md**
Document your project structure and conventions.

2. **Configure agents**
Create specialized agents for different tasks.

3. **Add PLAN.md**
For multi-agent coordination.

4. **Integrate MLflow**
For observability and tracking.

---

## Summary

OpenCode provides:

- Mature code operations (LSP, file management, multi-session)
- Programmatic control (run as server, control with Python)
- Flexible configuration (agents, tools, rules, skills)
- Multi-agent coordination (PLAN.md pattern)

Why it matters:

- Don't rebuild infrastructure
- Focus on your product
- Build custom workflows
- Production-ready from day one

The architecture is what makes it different. Not just another chat interface.

---

**Resources:**

- OpenCode docs: [opencode.ai](https://opencode.ai)
- Follow updates: [@AbdelkareemElk1]

---

**Related Posts:**

- [DSPy: Concepts and Exploration](../dspy/dspy.html)
- [Building SEO RAT: A Free, Open-Source SEO Tool](../seo/seo_rat_journey.html)
