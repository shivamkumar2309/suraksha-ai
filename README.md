---
title: SurakshaAI
colorFrom: pink
colorTo: blue
sdk: docker
app_port: 7860
---

# SurakshaAI - Women Safety AI Environment

## Overview

SurakshaAI is a real-world AI simulation environment where agents learn to make safety decisions using reward-based learning.

## Features

- 3 difficulty levels: easy, medium, hard  
- Real-world safety scenario simulation  
- Reward-based decision system  
- API-based environment interaction  

## Tasks

- Easy: clear danger (scream + unsafe)
- Medium: suspicious activity
- Hard: mixed signals (requires intelligent decision)

## Action Space

- send_alert  
- ignore  
- call_police  
- trigger_alarm  

## Observation Space

- time: day/night  
- location: safe/unsafe  
- sound: normal/scream  
- movement: normal/suspicious  

## Reward System

Reward is based on:
- danger level of the situation  
- correctness of the action  

## API Endpoints

### GET /
Check API status

### POST /reset
Reset environment  
Example: /reset?task=easy

### POST /step
Send action:
```json
{
  "action": "send_alert"
}