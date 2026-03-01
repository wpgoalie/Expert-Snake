class Fruit:
    def __init__(self, position):
        self.position = position  # grid coordinates (x, y)
        self.value = 1
        self.max_value = 1 # for normalization
        self.color = 'red'
        self.active = True

    def update(self):
        pass

    def on_eat(self):
        self.active = False
        return self.value


# fruit decays in value by 1 every decay_interval steps
class DecayFruit(Fruit):
    def __init__(self, position, initial_value=5, decay_interval=5):
        super().__init__(position)
        self.value = initial_value
        self.max_value = initial_value
        self.color = 'yellow'

        self.decay_interval = decay_interval
        self.timer = 0

    def update(self):
        self.timer += 1

        if self.timer >= self.decay_interval:
            self.value -= 1
            self.timer = 0

        if self.value <= 0:
            self.active = False