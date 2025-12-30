import os

# Defina seus vendedores aqui
VENDEDORES = ["Vendedor_Ana", "Vendedor_Bruno", "Vendedor_Carlos"]
FOLDERS = ["inputs", "outputs"]

def create_structure():
    print("👷‍♂️ Construindo QG da Neural Sales Ops...")
    
    for folder in FOLDERS:
        for vendedor in VENDEDORES:
            path = os.path.join(folder, vendedor)
            os.makedirs(path, exist_ok=True)
            print(f"✅ Pasta criada: {path}")

    print("\nestrutura pronta! Coloque os logs de chat dentro das pastas em 'inputs'.")

if __name__ == "__main__":
    create_structure()