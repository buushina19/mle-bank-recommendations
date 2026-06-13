from typing import Dict, List

from pydantic import BaseModel, Field


class ClientFeatures(BaseModel):
    ncodpers: int = Field(..., description="ID клиента")
    ind_empleado: str = "N"
    sexo: str = "H"
    age: float = 40.0
    ind_nuevo: int = 0
    antiguedad: float = 120.0
    indrel_1mes: str = "1"
    tiprel_1mes: str = "A"
    indresi: int = 1
    indext: int = 0
    conyuemp: int = 0
    canal_entrada: str = "KFC"
    ind_actividad_cliente: int = 1
    renta: float = 100000.0
    segmento: str = "02"
    products: Dict[str, int] = Field(
        default_factory=dict,
        description="Текущие продукты клиента: product_code -> 0/1",
    )


class RecommendationItem(BaseModel):
    product_code: str
    product_name: str
    score: float


class RecommendationResponse(BaseModel):
    ncodpers: int
    recommendations: List[RecommendationItem]
