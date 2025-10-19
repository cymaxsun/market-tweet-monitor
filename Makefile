.PHONY: backend frontend monitor dev stack lint clean

python ?= python3
uvicorn ?= uvicorn
npm ?= npm

backend:
	$(uvicorn) dashboard.backend:app --reload

frontend:
	cd dashboard/react-app && $(npm) run dev

monitor:
	$(python) tweet_monitor.py

stack:
	$(python) tweet_monitor.py & \
	$(uvicorn) dashboard.backend:app --reload & \
	cd dashboard/react-app && $(npm) run dev

lint:
	$(python) -m py_compile tweet_monitor.py dashboard/backend.py

clean:
	rm -rf .pycache __pycache__ dashboard/react-app/node_modules dashboard/react-app/dist

dev: stack
