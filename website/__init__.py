from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager

db = SQLAlchemy()
DB_NAME = "database.db"

#creating a flask application
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjhdj'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    #getting the blueprints by using relative import as we are inside an python package
    from .views import views
    from .auth import auth

    #registering the blueprints
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    #importing the models to create the database
    from .models import User

    #create database
    create_database(app)

    #flask login setup
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)


    #creating a session for the user
    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    


    return app

#in case the database doesn't exist, create it
def create_database(app):
    if not path.exists('website/' + DB_NAME):
        with app.app_context():
            db.create_all()
        print("Created Database!")