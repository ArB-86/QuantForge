class PreTradeRisk:

    def __init__(
        self,
        max_order_value=500000,
        max_quantity=10000,
    ):
        self.max_order_value = max_order_value
        self.max_quantity = max_quantity

    def validate(self, order):

        if order.quantity <= 0:
            raise ValueError("Invalid quantity")

        if order.quantity > self.max_quantity:
            raise ValueError("Quantity limit exceeded")

        if order.price is not None:
            value = order.quantity * order.price

            if value > self.max_order_value:
                raise ValueError("Order value limit exceeded")

        return True
