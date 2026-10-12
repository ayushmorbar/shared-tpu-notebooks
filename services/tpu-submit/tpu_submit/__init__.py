"""tpu-submit: the only component allowed to create TPU Jobs for students.

Students send a snapshot of their homework folder with their JupyterHub token. The
service builds the Job from a fixed template, so a student cannot skip the Kueue queue,
ask for more chips, pick another image, or borrow someone else's identity. It also owns
the only credential for the student bucket, which keeps every student's code and
checkpoints private to that student.
"""
