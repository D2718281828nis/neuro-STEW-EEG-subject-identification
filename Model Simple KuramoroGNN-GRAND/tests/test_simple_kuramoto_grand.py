"""
Comprehensive test suite for Simple KuramoroGNN-GRAND educational model.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to sys.path so we can import the main module
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_kuramoto_grand import (
    GrandConfig,
    GraphConfig,
    KuramotoConfig,
    compute_degree_matrix,
    compute_laplacian,
    dirichlet_energy,
    euler_integrate,
    grand_derivative,
    grand_message_matrix,
    kuramoto_derivative,
    kuramoto_message_matrix,
    kuramoto_order_parameter,
    normalize_adjacency,
    rk4_integrate,
    simulate_grand,
    simulate_kuramoto,
    validate_adjacency,
    validate_time_parameters,
    wrapped_phase_difference,
)

# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestAdjacencyValidation:
    """Test adjacency matrix validation."""
    
    def test_valid_adjacency(self):
        """Default adjacency should be valid."""
        graph = GraphConfig()  # Use default
        # Should not raise any exceptions
        validate_adjacency(graph.adjacency, graph.node_names)
    
    def test_square_requirement(self):
        """Adjacency must be square."""
        invalid = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
        with pytest.raises(ValueError, match="square"):
            validate_adjacency(invalid)
    
    def test_node_names_length(self):
        """Node names length must match matrix size."""
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        names = ("A", "B", "C")  # 3 names for 2x2 matrix
        with pytest.raises(ValueError, match="Node names length"):
            validate_adjacency(adj, names)
    
    def test_finite_values(self):
        """Adjacency must contain finite values."""
        adj = np.array([[0, np.inf], [np.inf, 0]], dtype=float)
        with pytest.raises(ValueError, match="non-finite"):
            validate_adjacency(adj)
    
    def test_symmetry(self):
        """Adjacency must be symmetric."""
        adj = np.array([[0, 1], [0.5, 0]], dtype=float)  # Not symmetric
        with pytest.raises(ValueError, match="symmetric"):
            validate_adjacency(adj)
    
    def test_nonnegative(self):
        """Adjacency weights must be nonnegative."""
        adj = np.array([[0, -1], [-1, 0]], dtype=float)
        with pytest.raises(ValueError, match="nonnegative"):
            validate_adjacency(adj)
    
    def test_zero_diagonal(self):
        """Diagonal must be zero (no self-loops)."""
        adj = np.array([[1, 1], [1, 0]], dtype=float)  # Nonzero diagonal
        with pytest.raises(ValueError, match="diagonal must be zero"):
            validate_adjacency(adj)


class TestTimeParameterValidation:
    """Test time parameter validation."""
    
    def test_valid_parameters(self):
        """Valid parameters should pass."""
        validate_time_parameters(0.01, 10.0)  # Should not raise
    
    def test_negative_dt(self):
        """Time step must be positive."""
        with pytest.raises(ValueError, match="dt must be positive"):
            validate_time_parameters(-0.01, 10.0)
    
    def test_zero_dt(self):
        """Time step must be positive (not zero)."""
        with pytest.raises(ValueError, match="dt must be positive"):
            validate_time_parameters(0.0, 10.0)
    
    def test_negative_duration(self):
        """Duration must be nonnegative."""
        with pytest.raises(ValueError, match="duration must be nonnegative"):
            validate_time_parameters(0.01, -1.0)


# =============================================================================
# GRAPH UTILITY TESTS
# =============================================================================

class TestGraphUtilities:
    """Test graph utility functions."""
    
    def test_degree_matrix(self):
        """Degree matrix should be diagonal with row sums."""
        adj = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
        deg = compute_degree_matrix(adj)
        
        # Should be diagonal matrix
        assert np.allclose(deg, deg.T)  # Symmetric
        assert np.allclose(np.diag(deg), [3, 4, 5])  # Row sums
        assert np.allclose(deg - np.diag(np.diag(deg)), 0)  # Diagonal
    
    def test_symmetric_normalization(self):
        """Symmetric normalization should preserve symmetry."""
        adj = np.array([[0, 1, 0.5], [1, 0, 0.2], [0.5, 0.2, 0]], dtype=float)
        norm_adj = normalize_adjacency(adj, "symmetric")
        
        # Should still be symmetric
        assert np.allclose(norm_adj, norm_adj.T)
        
        # Row sums should not necessarily be 1 (that's row normalization)
        # But should be properly normalized
    
    def test_row_normalization(self):
        """Row normalization should make each row sum to 1 (for connected nodes)."""
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        norm_adj = normalize_adjacency(adj, "row")
        
        # Each row should sum to 1 (for connected nodes)
        row_sums = np.sum(norm_adj, axis=1)
        assert np.allclose(row_sums[row_sums > 0], 1.0)
    
    def test_laplacian(self):
        """Test unnormalized Laplacian L = D - A."""
        adj = np.array([[0, 1, 0.5], [1, 0, 0.2], [0.5, 0.2, 0]], dtype=float)
        L = compute_laplacian(adj, normalized=False)
        deg = compute_degree_matrix(adj)
        expected_L = deg - adj
        
        assert np.allclose(L, expected_L)
    
    def test_normalized_laplacian(self):
        """Test normalized Laplacian L = I - D^(-1/2) A D^(-1/2)."""
        adj = np.array([[0, 1, 0.5], [1, 0, 0.2], [0.5, 0.2, 0]], dtype=float)
        L_norm = compute_laplacian(adj, normalized=True)
        expected = np.eye(3) - normalize_adjacency(adj, "symmetric")
        
        assert np.allclose(L_norm, expected)


# =============================================================================
# KURAMOTO FUNCTION TESTS
# =============================================================================

class TestKuramotoFunctions:
    """Test Kuramoto oscillator functions."""
    
    def test_message_matrix_shape(self):
        """Message matrix should be 3x3."""
        phases = np.array([0.0, 1.0, 2.0])
        adj = np.array([[0, 1, 0.5], [1, 0, 0.2], [0.5, 0.2, 0]], dtype=float)
        K = 2.0
        
        msg_matrix = kuramoto_message_matrix(phases, adj, K)
        assert msg_matrix.shape == (3, 3)
    
    def test_message_matrix_diagonal_zero(self):
        """Diagonal of message matrix should be zero (sin(0) = 0)."""
        phases = np.array([0.0, 1.0, 2.0])
        adj = np.array([[0, 1, 0.5], [1, 0, 0.2], [0.5, 0.2, 0]], dtype=float)
        K = 2.0
        
        msg_matrix = kuramoto_message_matrix(phases, adj, K)
        # Diagonal entries: messages[i,i] = K * A[i,i] * sin(θ_i - θ_i) = K * 0 * 0 = 0
        assert np.allclose(np.diag(msg_matrix), 0)
    
    def test_antisymmetric_messages(self):
        """For symmetric adjacency, messages should be antisymmetric: m_ij = -m_ji."""
        phases = np.array([0.0, 1.0, 2.0])
        adj = np.array([[0, 1, 0.5], [1, 0, 0.2], [0.5, 0.2, 0]], dtype=float)
        K = 2.0
        
        msg_matrix = kuramoto_message_matrix(phases, adj, K)
        # For symmetric adj: m_ij = K * A_ij * sin(θ_j - θ_i), m_ji = K * A_ji * sin(θ_i - θ_j) = K * A_ij * (-sin(θ_j - θ_i)) = -m_ij
        assert np.allclose(msg_matrix, -msg_matrix.T)
    
    def test_derivative_function(self):
        """Test kuramoto_derivative returns correct shapes."""
        phases = np.array([0.0, 1.0, 2.0])
        omega = np.array([1.0, 1.1, 0.9])
        adj = np.array([[0, 1, 0.5], [1, 0, 0.2], [0.5, 0.2, 0]], dtype=float)
        K = 2.0
        
        deriv, msg_matrix, coupling_terms = kuramoto_derivative(phases, omega, adj, K)
        
        assert deriv.shape == (3,)  # Phase derivatives for each node
        assert msg_matrix.shape == (3, 3)  # Message matrix
        assert coupling_terms.shape == (3,)  # Coupling term for each node
    
    def test_zero_coupling_derivative(self):
        """With zero coupling, derivative should equal natural frequencies."""
        phases = np.array([0.0, 1.0, 2.0])
        omega = np.array([1.0, 1.1, 0.9])
        adj = np.array([[0, 1, 0.5], [1, 0, 0.2], [0.5, 0.2, 0]], dtype=float)
        K = 0.0  # Zero coupling
        
        deriv, _, _ = kuramoto_derivative(phases, omega, adj, K)
        assert np.allclose(deriv, omega)
    
    def test_equal_phases_zero_coupling_messages(self):
        """When all phases are equal, coupling messages should be zero."""
        phases = np.array([1.0, 1.0, 1.0])  # All equal
        omega = np.array([1.0, 1.1, 0.9])
        adj = np.array([[0, 1, 0.5], [1, 0, 0.2], [0.5, 0.2, 0]], dtype=float)
        K = 2.0
        
        deriv, msg_matrix, coupling_terms = kuramoto_derivative(phases, omega, adj, K)
        
        # When all phases are equal, sin(θ_j - θ_i) = 0 for all i,j
        assert np.allclose(msg_matrix, 0)
        assert np.allclose(coupling_terms, 0)
        assert np.allclose(deriv, omega)  # Only natural frequency remains
    
    def test_order_parameter_identical_phases(self):
        """Order parameter should be 1 when all phases are identical."""
        phases = np.array([0.5, 0.5, 0.5])  # All identical
        r = kuramoto_order_parameter(phases)
        assert np.isclose(r, 1.0)
    
    def test_order_parameter_evenly_spaced(self):
        """Order parameter should be near zero for evenly spaced phases."""
        # Three phases evenly spaced around the circle
        phases = np.array([0, 2*np.pi/3, 4*np.pi/3])
        r = kuramoto_order_parameter(phases)
        assert r < 0.1  # Should be very small
    
    def test_wrapped_phase_difference_pi(self):
        """Test wrapped phase difference with [-π, π) convention."""
        # Test that 3π is wrapped to -π (3π - 2π = π, but wrapped to [-π, π) should be -π)
        diff = 3 * np.pi
        wrapped = wrapped_phase_difference(diff, "pi")
        expected = -np.pi  # 3π - 2π = π, then π - 2π = -π
        assert np.isclose(wrapped, expected)
    
    def test_wrapped_phase_difference_2pi(self):
        """Test wrapped phase difference with [0, 2π) convention."""
        diff = 3 * np.pi
        wrapped = wrapped_phase_difference(diff, "2pi")
        expected = np.pi  # 3π mod 2π = π
        assert np.isclose(wrapped, expected)


# =============================================================================
# GRAND FUNCTION TESTS
# =============================================================================

class TestGrandFunctions:
    """Test GRAND diffusion functions."""
    
    def test_message_matrix_shape(self):
        """Message matrix should be 3x3."""
        features = np.array([1.0, -0.5, 0.25])
        adj_norm = np.array([[0.5, 0.25, 0.1], [0.25, 0.4, 0.2], [0.1, 0.2, 0.8]], dtype=float)
        
        msg_matrix = grand_message_matrix(features, adj_norm)
        assert msg_matrix.shape == (3, 3)
    
    def test_derivative_function(self):
        """Test grand_derivative returns correct shapes."""
        features = np.array([1.0, -0.5, 0.25])
        adj_norm = np.array([[0.5, 0.25, 0.1], [0.25, 0.4, 0.2], [0.1, 0.2, 0.8]], dtype=float)
        
        deriv, msg_matrix, aggregated = grand_derivative(features, adj_norm)
        
        assert deriv.shape == (3,)  # Feature derivatives
        assert msg_matrix.shape == (3, 3)  # Message matrix
        assert aggregated.shape == (3,)  # Aggregated messages
    
    def test_constant_features_preserved(self):
        """If all features are the same, they should remain unchanged (derivative zero)."""
        features = np.array([1.0, 1.0, 1.0])
        adj_norm = normalize_adjacency(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float), "symmetric")
        
        deriv, _, aggregated = grand_derivative(features, adj_norm)
        
        # If all features are equal, aggregated should be same for all nodes
        # and derivative should be zero
        assert np.allclose(aggregated, aggregated[0] * np.ones(3))
        # Since X_i = X_j for all i,j, the derivative should be zero
        assert np.allclose(deriv, 0)
    
    def test_dirichlet_energy_zero(self):
        """Dirichlet energy should be zero when all features are equal."""
        features = np.array([1.0, 1.0, 1.0])
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        
        energy = dirichlet_energy(features, adj)
        assert np.isclose(energy, 0.0)
    
    def test_dirichlet_energy_positive(self):
        """Dirichlet energy should be positive when features differ."""
        features = np.array([1.0, 0.0, -1.0])
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        
        energy = dirichlet_energy(features, adj)
        assert energy > 0


# =============================================================================
# NUMERICAL INTEGRATION TESTS
# =============================================================================

class TestNumericalIntegration:
    """Test numerical integration functions."""
    
    def test_euler_integration_simple(self):
        """Test Euler integration with simple linear function."""
        def linear_deriv(x):
            return np.array([1.0])  # dx/dt = 1
        
        initial = np.array([0.0])
        times, trajectory = euler_integrate(linear_deriv, initial, 0.1, 1.0)
        
        # Should be approximately linear: x(t) = x(0) + t
        expected = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        assert np.allclose(trajectory.flatten(), expected, atol=1e-10)
    
    def test_euler_integration_shapes(self):
        """Test Euler integration returns correct shapes."""
        def simple_deriv(x):
            return np.array([1.0, -0.5])
        
        initial = np.array([0.0, 1.0])
        times, trajectory = euler_integrate(simple_deriv, initial, 0.1, 0.5)
        
        assert times.shape[0] == 6  # 0.5/0.1 + 1 = 6
        assert trajectory.shape == (6, 2)  # (n_steps, state_dim)
    
    def test_rk4_integration_simple(self):
        """Test RK4 integration with simple function."""
        def linear_deriv(x):
            return np.array([1.0])
        
        initial = np.array([0.0])
        times, trajectory = rk4_integrate(linear_deriv, initial, 0.1, 1.0)
        
        # Should still be approximately linear
        expected = np.linspace(0, 1.0, 11)
        assert np.allclose(trajectory.flatten(), expected, atol=1e-10)
    
    def test_rk4_more_accurate(self):
        """RK4 should be more accurate than Euler for nonlinear functions."""
        def quadratic_deriv(x):
            return np.array([x[0]**2])  # dx/dt = x^2
        
        initial = np.array([1.0])
        
        # Analytical solution: x(t) = 1/(1-t) for t < 1
        t = 0.5
        analytical = 1.0 / (1.0 - t)  # At t=0.5, x=2.0
        
        _, trajectory_euler = euler_integrate(quadratic_deriv, initial, 0.1, t)
        _, trajectory_rk4 = rk4_integrate(quadratic_deriv, initial, 0.1, t)
        
        euler_error = abs(trajectory_euler[-1, 0] - analytical)
        rk4_error = abs(trajectory_rk4[-1, 0] - analytical)
        
        # RK4 should be more accurate
        assert rk4_error < euler_error
    
    def test_invalid_dt(self):
        """Invalid dt should raise error."""
        def simple_deriv(x):
            return np.array([1.0])
        
        with pytest.raises(ValueError, match="dt must be positive"):
            euler_integrate(simple_deriv, np.array([0.0]), -0.1, 1.0)


# =============================================================================
# SIMULATION TESTS
# =============================================================================

class TestSimulations:
    """Test full simulation functions."""
    
    def test_kuramoto_simulation_basic(self):
        """Test basic Kuramoto simulation runs without errors."""
        graph = GraphConfig()
        config = KuramotoConfig(
            initial_phases=np.array([0.0, 1.0, 2.0]),
            natural_frequencies=np.array([1.0, 1.1, 0.9]),
            coupling_strength=1.0,
            dt=0.01, duration=0.1  # Short for testing
        )
        
        result = simulate_kuramoto(config, graph)
        
        assert result.times.shape[0] > 1  # Should have multiple time points
        assert result.phases.shape[1] == 3  # 3 nodes
        assert result.phase_velocities.shape == result.phases.shape
        assert result.message_matrices.shape == (len(result.times), 3, 3)
        assert result.order_parameters.shape == result.times.shape
    
    def test_kuramoto_simulation_all_finite(self):
        """Kuramoto simulation should produce finite values."""
        graph = GraphConfig()
        config = KuramotoConfig(
            initial_phases=np.array([0.0, 1.0, 2.0]),
            natural_frequencies=np.array([1.0, 1.1, 0.9]),
            coupling_strength=1.0,
            dt=0.01, duration=0.1
        )
        
        result = simulate_kuramoto(config, graph)
        
        assert np.all(np.isfinite(result.phases))
        assert np.all(np.isfinite(result.phase_velocities))
        assert np.all(np.isfinite(result.order_parameters))
    
    def test_grand_simulation_basic(self):
        """Test basic GRAND simulation runs without errors."""
        graph = GraphConfig()
        config = GrandConfig(
            initial_features=np.array([1.0, -0.5, 0.25]),
            dt=0.01, duration=0.1
        )
        
        result = simulate_grand(config, graph)
        
        assert result.times.shape[0] > 1
        assert result.features.shape[1] == 3
        assert result.feature_derivatives.shape == result.features.shape
        assert result.variances.shape == result.times.shape
        assert result.dirichlet_energies.shape == result.times.shape
    
    def test_grand_simulation_all_finite(self):
        """GRAND simulation should produce finite values."""
        graph = GraphConfig()
        config = GrandConfig(
            initial_features=np.array([1.0, -0.5, 0.25]),
            dt=0.01, duration=0.1
        )
        
        result = simulate_grand(config, graph)
        
        assert np.all(np.isfinite(result.features))
        assert np.all(np.isfinite(result.feature_derivatives))
        assert np.all(np.isfinite(result.variances))
        assert np.all(np.isfinite(result.dirichlet_energies))
    
    def test_grand_reduces_variance(self):
        """GRAND diffusion should reduce feature variance over time."""
        graph = GraphConfig()
        config = GrandConfig(
            initial_features=np.array([1.0, -1.0, 0.0]),  # High variance
            dt=0.01, duration=1.0
        )
        
        result = simulate_grand(config, graph)
        
        # Variance should generally decrease (though may have small fluctuations)
        initial_variance = result.variances[0]
        final_variance = result.variances[-1]
        assert final_variance < initial_variance
    
    def test_kuramoto_synchronization_increases(self):
        """With positive coupling, synchronization should increase."""
        graph = GraphConfig()
        config = KuramotoConfig(
            initial_phases=np.array([0.0, 2.0, 4.0]),
            natural_frequencies=np.array([1.0, 1.1, 0.9]),
            coupling_strength=5.0,  # Strong coupling
            dt=0.01, duration=2.0
        )
        
        result = simulate_kuramoto(config, graph)
        
        # Order parameter should increase over time with strong coupling
        initial_r = result.order_parameters[0]
        final_r = result.order_parameters[-1]
        assert final_r > initial_r
    
    def test_kuramoto_zero_coupling_no_sync(self):
        """With zero coupling, synchronization should stay low."""
        graph = GraphConfig()
        config = KuramotoConfig(
            initial_phases=np.array([0.0, 2.0, 4.0]),
            natural_frequencies=np.array([1.0, 1.1, 0.9]),
            coupling_strength=0.0,  # Zero coupling
            dt=0.01, duration=2.0
        )
        
        result = simulate_kuramoto(config, graph)
        
        # Order parameter should remain low (no synchronization without coupling)
        final_r = result.order_parameters[-1]
        assert final_r < 0.5  # Should remain dispersed


# =============================================================================
# END-TO-END TESTS
# =============================================================================

class TestEndToEnd:
    """End-to-end tests of the full system."""
    
    def test_cli_help_succeeds(self):
        """Test that --help command succeeds."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "simple_kuramoto_grand.py"), "--help"],
            capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
        )
        assert result.returncode == 0
        assert "Simple KuramoroGNN-GRAND" in result.stdout
    
    def test_quick_simulation_in_temp_dir(self):
        """Test that a quick simulation runs and generates expected files."""
        import subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable, 
                str(Path(__file__).parent.parent / "simple_kuramoto_grand.py"),
                "--quick", "--output-dir", tmpdir, "--duration", "0.5", "--dt", "0.05"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=str(Path(__file__).parent.parent))
            
            assert result.returncode == 0, f"Command failed: {result.stderr}"
            
            # Check that expected files were created
            expected_files = [
                "01_three_node_graph.png",
                "02_adjacency_matrix.png",
                "simulation_summary.json"
            ]
            
            for file in expected_files:
                assert (Path(tmpdir) / file).exists(), f"Expected file {file} not found"
            
            # Check JSON summary is valid
            with open(Path(tmpdir) / "simulation_summary.json") as f:
                summary = json.load(f)
                assert "args" in summary
                assert "graph" in summary
                assert "plots" in summary