BLUEPRINT 3D Robot Builder and Robot Testing Lab

1. Copy builder.html, builder.css and builder.js into D:\BLUEPRINT\frontend
2. Keep your existing index.html, style.css, script.js and app.py unchanged.
3. Start the server from D:\BLUEPRINT:
   .\.venv\Scripts\python.exe -m uvicorn app:app --reload
4. Open:
   http://127.0.0.1:8000/ui/builder.html
5. Add a chassis, a powered motor and a wheel, then use Testing Lab in the
   builder navigation. The current assembly configuration is saved in this
   browser's local storage for the lab to select:
   http://127.0.0.1:8000/ui/testing-lab.html

This assumes your app.py already mounts the frontend directory at /ui using FastAPI StaticFiles.
The browser needs internet access to load Three.js from jsDelivr.

Limits: the builder is a visual assembly prototype with simple geometry and does not enforce
joints or collisions. The Testing Lab reuses the existing simplified physics engine, with
terrain friction and rolling-resistance approximations and configurable gravity. Obstacle
traversal, wheel slip dynamics, center-of-gravity stability, chassis strength without supplied
properties, component environmental qualification, battery ageing, and thermal lifetime remain
unsupported and are explicitly marked NOT EVALUATED. Test history and CSV export are stored in
the current browser; the project does not currently generate PDF reports or claim laboratory
qualification.
