import torch
import numpy as np

def print_tensor_info(name, tensor):
    print(f"--- Tensor: {name} ---")
    print(f"Shape: {tensor.shape}")
    print(f"Stride: {tensor.stride()}")
    print(f"Is Contiguous: {tensor.is_contiguous()}")
    print(f"Storage Data Ptr: {tensor.storage().data_ptr()}")
    # print(f"First Element Address: {tensor.data_ptr()}") # Same as storage for simple tensors
    print(f"Content:\n{tensor}")
    print("-" * 30)

def demonstrate_stride():
    print("=== 1. Creating a 3x4 Tensor ===")
    # Create a 3x4 tensor
    A = torch.arange(12).reshape(3, 4)
    print_tensor_info("A", A)

    print("\n=== 2. Transposing (Logic Change Only) ===")
    # Transpose
    B = A.t()
    print_tensor_info("B = A.t()", B)
    
    print("\n=== 3. Storage Analysis ===")
    print(f"A storage ptr: {A.storage().data_ptr()}")
    print(f"B storage ptr: {B.storage().data_ptr()}")
    if A.storage().data_ptr() == B.storage().data_ptr():
        print(">> VERIFIED: A and B share the SAME physical memory!")
    else:
        print(">> DIFFERENT memory!")

    print("\n=== 4. The View Trap ===")
    try:
        # This will fail because B is not contiguous
        C = B.view(12) 
    except RuntimeError as e:
        print(f">> ERROR CAUGHT: {e}")
        print(">> Explanation: B is column-major (stride=(1,4)), but view() expects row-major.")

    print("\n=== 5. Making Contiguous (Copying Data) ===")
    B_contig = B.contiguous()
    print_tensor_info("B_contig", B_contig)
    print(f"B_contig storage ptr: {B_contig.storage().data_ptr()}")
    if B_contig.storage().data_ptr() != A.storage().data_ptr():
         print(">> VERIFIED: B_contig has NEW physical memory (Copy happened).")

def demonstrate_broadcasting():
    print("\n\n=== 6. Broadcasting Magic ===")
    # A (3, 1)
    a = torch.tensor([[1], [2], [3]])
    print_tensor_info("a (3, 1)", a)

    # Broadcast to (3, 3)
    # expand() creates a view, does not copy data
    b = a.expand(3, 3)
    print_tensor_info("b = a.expand(3, 3)", b)
    
    print(f"a storage ptr: {a.storage().data_ptr()}")
    print(f"b storage ptr: {b.storage().data_ptr()}")
    
    # Check stride 0
    print(f"b stride: {b.stride()}")
    print(">> Notice the stride is (1, 0)! The 0 stride means moving in that dimension costs nothing in memory.")

if __name__ == "__main__":
    demonstrate_stride()
    demonstrate_broadcasting()
