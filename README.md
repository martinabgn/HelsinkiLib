# Library & Services Finder for Helsinki Metropolitan Area

## Overview
This application simplifies the search for libraries, services and events across **Helsinki, Espoo, and Vantaa**. Users can quickly find the libraries **around them**, or find the ones which provide various services such as **3D printing, sewing machines, and group workspaces**, or **events**.

## Problem Statement
HelsinkiLib aims to simplify the process of finding libraries, services, and events in Helsinki. With a vast number of libraries offering various resources, users often struggle to locate relevant services, upcoming events, or even basic library information. This app provides a centralized search tool that allows users to efficiently explore library-related data using different search methods, ensuring quick and accurate access to the information they need. 

## Features
- **Search for libraries** by name or location.
- **Find services** (e.g., 3D printing, sewing machines, group study rooms).
- **Access library chain websites** directly from the application.
- **Look for the events** held by Helsinki's libraries.

## Installation
The project directory will be the cloned repository:
```bash
git clone https://github.com/martinabgn/HelsinkiLib.git
cd HelsinkiLib
```
(OPTIONAL) work in a virtual environment `myenv`:
```bash
python3 -m venv myenv
. myenv/bin/activate
```
Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Start searching!
```bash
export FLASK_APP=hl1.py
export FLASK_DEBUG=True
export FLASK_RUN_PORT=8800
flask run
```
