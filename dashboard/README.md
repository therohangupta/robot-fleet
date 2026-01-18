# Robot Fleet Dashboard

A modern web interface for managing and monitoring your robot fleet.

![Dashboard Preview](https://via.placeholder.com/800x400/0d1117/00d9ff?text=Robot+Fleet+Dashboard)

## Features

- 🤖 **Robot Management** - Register, monitor, and manage your robot fleet
- 🎯 **Goal Management** - Create and track high-level goals
- 📋 **Plan Builder** - Visual plan creation with strategy selection
- 🔄 **Live Execution** - Real-time DAG visualization of plan execution
- 🌍 **World State** - Manage environment state descriptions

## Tech Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **gRPC Bridge** - Connects to the Fleet Manager service

### Frontend
- **React 18** with TypeScript
- **Vite** - Fast build tool
- **TailwindCSS** - Utility-first styling
- **TanStack Query** - Data fetching and caching
- **React Flow** - DAG visualization
- **Framer Motion** - Animations

## Quick Start

### Prerequisites

1. **Fleet Manager server** must be running:
   ```bash
   python -m robot_fleet.server
   ```

2. **Node.js 18+** for the frontend

### Start the Dashboard

**Terminal 1 - Backend:**
```bash
cd dashboard
chmod +x run_backend.sh
./run_backend.sh
```

**Terminal 2 - Frontend:**
```bash
cd dashboard
chmod +x run_frontend.sh
./run_frontend.sh
```

Then open http://localhost:5173 in your browser.

### Manual Start

**Backend:**
```bash
cd dashboard
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd dashboard/frontend
npm install
npm run dev
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Browser (localhost:5173)                     │
│                         React Frontend                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP / WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Dashboard Backend (localhost:8000)             │
│                         FastAPI Server                          │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ REST API    │  │ WebSocket   │  │ gRPC Bridge             │ │
│  │ /api/*      │  │ /ws/*       │  │ FleetManagerClient      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ gRPC
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                Fleet Manager (localhost:50051)                  │
│                      gRPC Service                               │
└─────────────────────────────────────────────────────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/robots` | GET | List all robots |
| `/api/robots/register` | POST | Register a new robot |
| `/api/robots/{id}` | DELETE | Unregister a robot |
| `/api/goals` | GET, POST | List/create goals |
| `/api/goals/{id}` | DELETE | Delete a goal |
| `/api/plans` | GET, POST | List/create plans |
| `/api/plans/{id}/start` | POST | Start plan execution |
| `/api/tasks` | GET | List tasks |
| `/api/world` | GET, POST | List/add world statements |
| `/api/strategies` | GET | Get available strategies |
| `/ws/execution/{plan_id}` | WebSocket | Live execution updates |

## Development

### Project Structure

```
dashboard/
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── models.py        # Pydantic models
│   └── grpc_bridge.py   # gRPC client wrapper
├── frontend/
│   ├── src/
│   │   ├── components/  # Reusable UI components
│   │   ├── pages/       # Page components
│   │   ├── lib/         # API client, utilities
│   │   ├── App.tsx      # Main app with routing
│   │   └── main.tsx     # Entry point
│   ├── package.json
│   └── vite.config.ts
├── requirements.txt
├── run_backend.sh
├── run_frontend.sh
└── README.md
```

### Customization

- **Theme**: Edit `tailwind.config.js` for colors and fonts
- **API**: Modify `frontend/src/lib/api.ts` for endpoint changes
- **Components**: Add new components in `frontend/src/components/`

## License

Part of the Robot Fleet project.
