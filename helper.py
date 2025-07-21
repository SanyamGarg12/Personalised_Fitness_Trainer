import os

# List of Surya Namaskar steps
steps = [
    "1. Pranamasana (Prayer pose)",
    "2. Hastauttanasana (Raised arms pose)",
    "3. Hasta Padasana (Hand to foot pose)",
    "4. Ashwa Sanchalanasana (Equestrian pose)",
    "5. Dandasana (Stick pose)",
    "6. Ashtanga Namaskara (Salute with eight parts)",
    "7. Bhujangasana (Cobra pose)",
    "8. Adho Mukha Svanasana (Downward dog pose)",
    "9. Ashwa Sanchalanasana (Equestrian pose)",
    "10. Hasta Padasana (Hand to foot pose)",
    "11. Hastauttanasana (Raised arms pose)",
    "12. Pranamasana (Prayer pose)"
]

# Directory where folders will be created
base_dir = "SuryaNamaskar_Steps"
os.makedirs(base_dir, exist_ok=True)

# Create folders for each step
for step in steps:
    folder_name = os.path.join(base_dir, step)
    try:
        os.makedirs(folder_name, exist_ok=True)
        print(f"Created: {folder_name}")
    except Exception as e:
        print(f"Error creating {folder_name}: {e}")
