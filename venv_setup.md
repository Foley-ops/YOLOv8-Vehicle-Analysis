# Setting Up a Python Virtual Environment on Mac

Follow these steps to set up a Python virtual environment on your Mac and install necessary libraries for your project.

## 1. Open Terminal
- Press `Cmd + Space`, type **Terminal**, and hit **Enter**.

## 2. Navigate to Your Project Folder
- Use the `cd` command to go to your project directory:
  ```bash
  cd path/to/your/project
  ```

## 3. Create a Virtual Environment
- Run the following command to create a virtual environment named `venv`:
  ```bash
  python3 -m venv venv
  ```

## 4. Activate the Virtual Environment
- Activate the virtual environment with:
  ```bash
  source venv/bin/activate
  ```
- You’ll see `(venv)` appear in your terminal, indicating the environment is active.

## 5. Install Required Libraries
- Use `pip` to install necessary libraries, e.g., `matplotlib`:
  ```bash
  pip install matplotlib
  ```

## 6. Deactivate the Virtual Environment
- Once you're done, deactivate the environment with:
  ```bash
  deactivate
  ```

You're now ready to run your Python scripts within this virtual environment!

