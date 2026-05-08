---
title: Hacker AI Engineer Interview 2026 Example
description: HackerRank AI engineer interview review 2026. prompt engineering, SQL, and RAG coding challenges that test memorization over real-world AI skills.
author: kareem
date: 2026-04-28
draft: false
featured: false
categories:
  - blogging
  - til
  - blog/opinion/tech 
image: images/hackerank.jpeg
---

## HackerRank AI Engineer Interview

I was applying to a company for a junior AI role.

I'm not here to talk about the company or the role, but about the idea of using a dumb system like HackerRank to assess your knowledge.

The interview was just 3 questions, and the time limit was 2 hours and 40 minutes.

### Prompt Engineering

It's a question about creating a prompt for an LLM to do some finance equations, to test if the model will answer the user correctly.

It's a very simple and easy question, but the problem is:

what model are they using to follow the prompt?

There are 6 questions.

4 of them need the output to be just a number, and 2 need it to be in the following format:

- Final Price: $42, Discount Price: $23

When you try to instruct the LLM to do this, it fails.

I don't think there is room to improve this, the model is very small and doesn't follow instructions correctly.

This doesn't assess how much knowledge you have about small LLMs and how to use them.

**That's a different story.**

### SQL Question

A SQL question about using aggregations and aliases, a medium-difficulty question.

But the hard part was that the answer shouldn't be in table format and should be tab-separated.

As someone who has been working with SQL through ORMs for all my use cases, I had never encountered this case, so I spent around an hour figuring out how to do it.

I don't think this assesses anything more than just syntax.

### RAG Built with LangChain, LangGraph, and FAISS

Two Python files where you need to fill in the missing methods.

The problem here is that to write these methods, you must remember the exact syntax for these specific tools — which I don't even use.

I use DSPy and Qdrant for most of my work.
I've used FAISS before, but I'm not going to memorize how to import its modules.

### Final Thoughts

Using HackerRank for this type of role, if it shows anything, it shows that the people running the interview don't know much about AI engineering.

read more :
[LiteRT and Qualcomm AI Hub](https://kareemai.com/blog/posts/on_device/litert.html)
[Vector Databases Book O'reilly](https://kareemai.com/blog/posts/nlp/embedding_world/vector_database_book.html)
[Abdelkareem Elkhateb](https://kareemai.com/)
[HackerRank](https://www.hackerrank.com/)
