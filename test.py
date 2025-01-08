from src.scripts_bullet.CoordinateManager import CoordinateManager

def main():
    # Erstelle eine Instanz von CoordinateManager (Singleton)
    manager = CoordinateManager()

    # Test 1: Setzen und Abrufen eines flachen Arrays
    print("Testing with flat array...")

    flat_array = [1.0, 2.0, 3.0, 4.0]
    manager.set_coordinates(flat_array)
    retrieved_coordinates = manager.get_coordinates()
    print("Set:", flat_array)
    print("Retrieved:", retrieved_coordinates)

if __name__ == "__main__":
    main()
