import jax

def configure_jax() -> None:
    jax.config.update("jax_enable_x64", True)