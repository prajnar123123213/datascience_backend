from datetime import datetime
from sqlalchemy.exc import IntegrityError
from __init__ import app, db  # your Flask app and SQLAlchemy instance

class TitanicPrediction(db.Model):
    """
    TitanicPrediction Model
    
    Stores prediction results for Titanic passengers made by users.
    
    Attributes:
        id (int): Primary key.
        user_id (int): Foreign key to User who made the prediction.
        input_data (JSON): Passenger data input for prediction.
        prediction_result (JSON): Result of the prediction.
        created_at (datetime): Timestamp of creation.
    """
    __tablename__ = 'titanic_predictions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    input_data = db.Column(db.JSON, nullable=False)
    prediction_result = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship to user
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

    def __init__(self, user_id, input_data, prediction_result):
        """
        Constructor for TitanicPrediction
        
        Args:
            user_id (int): ID of the user who made the prediction.
            input_data (dict): The passenger input data used for prediction.
            prediction_result (dict): The prediction results.
        """
        self.user_id = user_id
        self.input_data = input_data
        self.prediction_result = prediction_result

    def __repr__(self):
        return f"<TitanicPrediction id={self.id} user_id={self.user_id} created_at={self.created_at}>"

    def create(self):
        """
        Save this TitanicPrediction instance to the database.
        
        Returns:
            TitanicPrediction: self if successful, None otherwise.
        """
        try:
            db.session.add(self)
            db.session.commit()
            return self
        except IntegrityError as e:
            db.session.rollback()
            print(f"IntegrityError on create: {e}")
            return None

    def delete(self):
        """
        Delete this TitanicPrediction instance from the database.
        """
        try:
            db.session.delete(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e

    def to_dict(self):
        """
        Convert the TitanicPrediction instance into a dictionary.
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "input_data": self.input_data,
            "prediction_result": self.prediction_result,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }