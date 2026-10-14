---
permalink: "/til/tils/livekitintro"
section: "til"
title: ماهي Likve Kit مقدمة سريعة
description: هي مشروع مفتوح المصدر لبناء وكلاء ذكاء إصطناعي تتعامل مع الصوت و الصورة
  بشكل لحظي وتشمل كل ما يحتاجه المبرمج
lang: ar
dir: rtl
author: kareem
publish: true
date: '2026-07-05'
modified: '2026-07-25'
tags:
- projects
- note
- blogging
- til
- blog/opinion/tech
socialImage: til.jpg
---

## اي هي Live Kit?

- مشروع مفتوح المصدر بتستخدمه Openai علشان تبني voice system بتاعها!
- ده يدل علي مدي أهمية المشروع والموثوقية الي بيحملها.

![LiveKit Intro](Pasted%20image%2020260705230522.png)

- تقدر تبني Vocie AI و تدمج المشروع مع خدمات الموبايل
- بيجي معاها full-stack observability متكامل علشان تدير النموذج الخاص بيك بشكل سلسل.

## LivekIt بتشتغل ازاي؟

بتتكون من أكثر من جزء لكن بإختصار:

1. المتحدث بيكلم النموذج من خلال تطبيق موبايل أو المتصفح أو من مكالمة موبايل عاديه.
2. الكلام بيتم إيصاله بشكل فوري جزء بجزء للنموذج Streaming
3. النموذج بيقبل الكلام ويبدا يرد بناء علي البيزنس بتاعك والطريقة الي انت برمجته يرد بيها

### LiveKit بتفصيل أكثر

- **بتستخدم بروتوكول WebRTC**
- البروتوكول ده مخصص إنه :
  1. ينقل الصوت والفيديو بشكل فوري بيختلف تمام عن http
  2. بيقدر يتعامل مع الشبكات الردئية والإنقطاعات والتذبذب في الشبكات
  3. بيشتغل علي اي متصفح

     المشروع بيقدر انه يستخدمه علشان تبني فوق البر.وتوكول المميزات الي اتكملنا عنها فوق.

### مقالات قد تكون مفيده

- [أخر تطورات TTS](https://kareemai.com/blog/posts/nlp/TTS%20Architecture%20V1.html)
- [Hacker Rank Interview 2026 for AI](https://kareemai.com/blog/posts/ml_life/hacker_rank_ai_intereviw.html)
