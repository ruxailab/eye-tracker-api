class Session:
    """
    Represents a session in the eye tracking application.

    Attributes:
        - id (int): The unique identifier of the session.
        - title (str): The title of the session.
        - description (str): The description of the session.
        - user_id (int): The user ID associated with the session.
        - created_date (datetime): The date and time when the session was created.
        - website_url (str): The URL of the website being tracked.
        - screen_record_url (str): The URL of the screen recording for the session.
        - webcam_record_url (str): The URL of the webcam recording for the session.
        - heatmap_url (str): The URL of the heatmap image for the session.
        - calib_points (list): The calibration points used in the session.
        - iris_points (list): The iris tracking points recorded in the session.
    """

    def __init__(
        self,
        id,
        title,
        description,
        user_id,
        created_date,
        website_url,
        screen_record_url,
        webcam_record_url,
        heatmap_url,
        calib_points,
        iris_points,
    ):
        self.id = id
        self.title = title
        self.description = description
        self.user_id = user_id
        self.created_date = created_date
        self.website_url = website_url
        self.screen_record_url = screen_record_url
        self.webcam_record_url = webcam_record_url
        self.heatmap_url = heatmap_url
        self.calib_points = calib_points
        self.iris_points = iris_points

    def to_dict(self):
        """
        Converts the session object to a dictionary.

        Returns:
            dict: A dictionary representation of the session object.
        """
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "user_id": self.user_id,
            "created_date": self.created_date,
            "website_url": self.website_url,
            "screen_record_url": self.screen_record_url,
            "webcam_record_url": self.webcam_record_url,
            "heatmap_url": self.heatmap_url,
            "calib_points": self.calib_points,
            "iris_points": self.iris_points,
        }
