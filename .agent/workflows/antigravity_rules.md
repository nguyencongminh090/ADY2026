---
description: Antigravity Code Rules and Instructions
---
# Antigravity Rules

As an AI Agent assisting the user with the `PHProject`, you must strictly adhere to the following rules at all times.

## 1. Safety and Reliability
- **Do Not Break the Project**: Before making any modification, carefully read and understand the existing codebase (e.g., `src/preprocess.py`, data pipelines). Ensure that your changes do not introduce regressions or break existing functionality.
- **Strictly Follow User Requirements**: Do not hallucinate features or make assumptions. Implement exactly what the user asks for. Ask for clarification if requirements are ambiguous.

## 2. Code Quality
- **Clean Code**: Adhere to Python PEP 8 standards and language-specific best practices. Use meaningful variable and function names.
- **Maintainability**: Write modular, self-explaining, and easily maintainable code. Add concise, informative docstrings and inline comments where complex logic is involved. Keep functions small and focused on a single responsibility.

## 3. Software Architecture
- **Clean Design**: Employ a robust software architecture. Separate concerns logically (e.g., keep data loading, preprocessing, and model training in distinct modules or functions).
- **UML Diagrams**: When designing new features, proposing significant changes, or documenting the existing system, you must design the structure and provide UML diagrams. Use Markdown Mermaid syntax to generate:
  - **Class Diagrams**: To show object-oriented structures and system components.
  - **State Diagrams**: To illustrate the lifecycle of objects, application states, or processes.
  - **Data Flow Diagrams**: To trace how data moves through the pipeline (e.g., from `data/raw/` through `src/preprocess.py` to `data/processed/`).
- **Scalability**: Ensure new structures plug cleanly into the existing ecosystem and allow for easy future expansion.
