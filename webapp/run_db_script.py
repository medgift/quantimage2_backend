# from flask import Flask
# from flask_sqlalchemy import SQLAlchemy

# # Initialize the Flask application
# app = Flask(__name__)

# # Configure the SQLAlchemy part of the app instance
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://quantimage2:nX1a5QIfucYBODHfYDcz01MjlMFRoUsdug5k4RvaBj0=@localhost:3307/quantimage2'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# # Create the SQLAlchemy db instance
# db = SQLAlchemy(app)

# # Push an application context
# ctx = app.app_context()
# ctx.push()

from sqlalchemy import create_engine
import pandas as pd

# Create an engine instance
engine = create_engine('mysql+pymysql://quantimage2:nX1a5QIfucYBODHfYDcz01MjlMFRoUsdug5k4RvaBj0=@localhost:3307/quantimage2')

# Connect to the database
conn = engine.connect()

# Define your SQL query
query = "SELECT * FROM feature_collection"

# Execute the query and load the data into a DataFrame
df = pd.read_sql_query(query, conn)

breakpoint()

print(df)