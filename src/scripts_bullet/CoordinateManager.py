class CoordinateManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CoordinateManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._coordinates = []  # Initialisiere hier
        return cls._instance

    def get_coordinates(self):
        return self._coordinates

    def set_coordinates(self, coordinates):
        if isinstance(coordinates, list):
            # Überprüfen, ob die Liste gültig ist (flach oder Paare)
            if all(isinstance(coord, (int, float)) for coord in coordinates):
                self._coordinates = coordinates
            elif all(isinstance(coord, (tuple, list)) and len(coord) == 2 for coord in coordinates):
                self._coordinates = coordinates
            else:
                raise ValueError("Input must be a flat list of numbers or a list of coordinate pairs.")
        else:
            raise ValueError("Coordinates must be a list.")
