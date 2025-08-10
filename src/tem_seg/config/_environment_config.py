from pydantic import BaseModel, Field


class EnvironmentConfig(BaseModel):
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    tf_cpp_min_log_level: str = Field(
        default="2",
        description="Tensorflow log level",
    )
