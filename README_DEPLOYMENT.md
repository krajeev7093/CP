# Deployment Guide

This Streamlit app is ready for Streamlit Community Cloud.

## Files to keep in the GitHub repo

- `app.py`
- `requirements.txt`
- `mobile_dataset.xlsx`
- `laptop_dataset.xlsx`
- `drone_dataset.xlsx`

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to `https://share.streamlit.io`.
3. Click `Create app`.
4. Choose `Yup, I have an app`.
5. Select your GitHub repository and branch.
6. Set the app file path to `app.py`.
7. In advanced settings, keep Python as `3.12` unless you have a reason to change it.
8. Click `Deploy`.

After deployment, Streamlit will give you a public `streamlit.app` URL.

## Local test before deploying

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Common deployment fixes

- If deployment says a module is missing, confirm it is listed in `requirements.txt`.
- If Excel files are not found, confirm all three `.xlsx` files are committed to GitHub in the same folder as `app.py`.
- If the app deploys but charts or tables look old, reboot the app from Streamlit Cloud's app menu.
