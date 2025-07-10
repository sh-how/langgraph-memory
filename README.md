### Why memory for AI agents?

https://blog.langchain.com/context-engineering-for-agents/

Memory management is vital when it comes to AI agents especially AI agents in multi-agent architectures

Blog - [[https://blog.langchain.com/the-rise-of-context-engineering/]]

The reason AI agents fail at their task boils down to two primary reasons:
 - Underlying model is not good enough
 - The underlying model was not passed the appropriate context to provide a good output

The focus of memory for AI agents aims to solve the second issue of passing context that is necessary for the AI agent to accomplish its tasks correctly

Context that is passed to the AI agents can be bad for a few reasons:
 - Missing context, forcing the AI agent to hallucinate or come up with context that was otherwise not provided or missing
 - Poorly formatted context 
