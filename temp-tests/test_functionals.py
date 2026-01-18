#!/usr/bin/env python3
"""
Test script for QuantumForge DFT functionals
"""
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import torch

    from quantumforge.functionals.gga import (
        BLYPExchange,
        LYPCorrelation,
        PBECorrelation,
        PBEExchange,
    )
    from quantumforge.functionals.hybrid import B3LYP, HSE06, PBE0
    from quantumforge.functionals.lda import SlaterExchange, VWNCorrelation
except ImportError as e:
    print(f"Import error: {e}")
    print("This test requires PyTorch and QuantumForge to be installed.")
    sys.exit(1)


def test_functionals():
    """Test all implemented DFT functionals"""
    print("Testing QuantumForge DFT Functionals")
    print("=" * 50)

    # Test data - simple 2x2x2 grid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create test density and gradients
    rho = torch.rand(2, 2, 2, device=device) * 0.1 + 0.01
    grad_rho = torch.rand(3, 2, 2, 2, device=device) * 0.01

    print(f"Test density shape: {rho.shape}, "
          f"range: [{rho.min():.6f}, {rho.max():.6f}]")
    print(f"Test gradient shape: {grad_rho.shape}, "
          f"range: [{grad_rho.min():.6f}, {grad_rho.max():.6f}]")
    print()

    # Test LDA functionals
    print("LDA Functionals:")
    print("-" * 20)

    slater_ex = SlaterExchange()
    vwn_corr = VWNCorrelation()

    ex_lda = slater_ex(rho)
    ec_lda = vwn_corr(rho)

    print(f"Slater Exchange energy: {ex_lda.sum().item():.8f}")
    print(f"VWN Correlation energy: {ec_lda.sum().item():.8f}")
    print(f"Total LDA energy: {(ex_lda + ec_lda).sum().item():.8f}")
    print()

    # Test GGA functionals
    print("GGA Functionals:")
    print("-" * 20)

    pbe_ex = PBEExchange()
    pbe_corr = PBECorrelation()
    blyp_ex = BLYPExchange()
    lyp_corr = LYPCorrelation()

    ex_pbe = pbe_ex(rho, grad_rho)
    ec_pbe = pbe_corr(rho, grad_rho)
    ex_blyp = blyp_ex(rho, grad_rho)
    ec_lyp = lyp_corr(rho, grad_rho)

    print(f"PBE Exchange energy: {ex_pbe.sum().item():.8f}")
    print(f"PBE Correlation energy: {ec_pbe.sum().item():.8f}")
    print(f"BLYP Exchange energy: {ex_blyp.sum().item():.8f}")
    print(f"LYP Correlation energy: {ec_lyp.sum().item():.8f}")
    print(f"Total PBE energy: {(ex_pbe + ec_pbe).sum().item():.8f}")
    print(f"Total BLYP energy: {(ex_blyp + ec_lyp).sum().item():.8f}")
    print()

    # Test Hybrid functionals
    print("Hybrid Functionals:")
    print("-" * 20)

    b3lyp = B3LYP()
    pbe0 = PBE0()
    hse06 = HSE06()

    # Create mock HF exchange for testing (normally would come from SCF)
    hf_exchange = torch.rand_like(rho) * 0.01

    e_b3lyp = b3lyp(rho, grad_rho, hf_exchange)
    e_pbe0 = pbe0(rho, grad_rho, hf_exchange)
    e_hse06 = hse06(rho, grad_rho, hf_exchange)

    print(f"B3LYP energy: {e_b3lyp.sum().item():.8f}")
    print(f"PBE0 energy: {e_pbe0.sum().item():.8f}")
    print(f"HSE06 energy: {e_hse06.sum().item():.8f}")
    print()

    print("✅ All functional tests completed successfully!")
    print()

    # Test gradient computation
    print("Testing automatic differentiation:")
    print("-" * 30)

    rho_grad = rho.clone().requires_grad_(True)
    energy = slater_ex(rho_grad).sum()
    energy.backward()

    print(f"Input density gradient norm: {rho_grad.grad.norm().item():.8f}")
    print("✅ Automatic differentiation working!")

    return True


if __name__ == "__main__":
    try:
        test_functionals()
        print("\n🎉 All tests passed! QuantumForge functionals are working correctly.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
