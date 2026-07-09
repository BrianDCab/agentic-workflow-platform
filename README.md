# Agentic Workflow Platform

**Live Demo:** https://agentic-workflow-platform.streamlit.app  
**GitHub:** https://github.com/BrianDCab/agentic-workflow-platform  
**Portfolio:** https://briancabrera.io  

> **Reviewer note:** If you were linked to this repository from a job application portal, it may be because the portal did not accept a valid `.streamlit.app` URL. The live deployed version is available here:  
> https://agentic-workflow-platform.streamlit.app

---

## Overview

The **Agentic Workflow Platform** is a Python and Streamlit application that turns uploaded CSV or Excel files into a guided analytics workflow.

The goal of this project was not to build a basic chatbot. I wanted to build something closer to a practical internal data tool: a system that can upload data, validate columns, clean messy values, segment records, generate charts, recommend actions, and keep the process reviewable.

This project demonstrates data engineering, analytics automation, internal tooling, AI-assisted decision support, and human-in-the-loop workflow design.

---

## What It Does

The app supports multiple analysis modes:

- **Players / Customers**
- **Companies / Accounts**
- **Custom / Any File**
- **Experimental Autonomous Agent Mode**

Users can upload a CSV or Excel file, map key columns, run segmentation logic, view charts and tables, generate recommendations, and export outputs.

---

## Key Features

- CSV and Excel file upload
- Custom column mapping for ID, value, and recency fields
- Data cleaning and validation
- Player/customer segmentation
- Company/account segmentation
- Custom file segmentation
- Value and recency-based analysis
- Data health checks
- Segment summaries and visualizations
- Per-segment drilldowns
- Individual record lookup
- AI-generated recommendations based on the user’s goal
- Sidebar chat for asking questions about the analyzed data
- Exportable CSV and text summaries
- Experimental autonomous mode where the agent chooses analysis steps
- Optional human approval before agent actions run
- AI provider fallback handling

---

## Why I Built This

I built this project to practice and demonstrate how data workflows can be made more usable, reliable, and reviewable.

Many analytics tasks involve the same pattern:

1. Upload or receive messy data
2. Identify important columns
3. Clean and validate values
4. Segment or classify records
5. Generate summaries and charts
6. Recommend next actions
7. Export results for business use

This project turns that pattern into an interactive workflow with validation and human review points instead of relying on a black-box response.

---

## Tech Stack

- Python
- Streamlit
- pandas
- Altair
- Groq API
- Gemini API fallback
- Google Generative AI SDK
- python-dotenv
- openpyxl
- CSV and Excel workflows

---

## Example Use Cases

### Casino / Customer Analytics

Upload player or customer data and identify:

- High-value active customers
- High-value customers at risk of churn
- Lower-priority casual segments
- Offer or comp recommendations
- Host worklists
- Value concentration by segment

### Business Account Analysis

Upload account data and identify:

- Strategic accounts
- High-value accounts at risk
- Leveraged watchlist accounts
- Dormant accounts
- Revenue and risk patterns

### Custom File Analysis

Upload almost any structured CSV or Excel file and map:

- An ID column
- A value column
- An optional recency column

The tool then builds generic value-based or value-plus-recency segments.

---

## Workflow

```text
Upload Data
   ↓
Preview File
   ↓
Map Columns
   ↓
Clean and Validate Data
   ↓
Segment Records
   ↓
Generate Charts and Tables
   ↓
Create Recommendations
   ↓
Review Individual Records
   ↓
Export Results
