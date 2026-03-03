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

# fruit that goes in a square path and has a penalty of 1 if eaten
class EnemyFruit(Fruit):
    def __init__(self, position, initial_value=-1, path_length = 5):
        super().__init__(position)
        self.value = initial_value
        self.max_value = initial_value
        self.color = 'purple'
        self.path_length = path_length
        self.cur_side = 0
        self.count = 0

    def update(self):
        if self.cur_side == 0:
            if self.count < self.path_length:
                self.position[0] += 1 # moving right
                self.count += 1
            else:
                self.cur_side = (self.cur_side + 1) % 4
                self.count = 0
        elif self.cur_side == 1:
            if self.count < self.path_length:
                self.position[1] += 1 # moving down
                self.count += 1
            else:
                self.cur_side = (self.cur_side + 1) % 4
                self.count = 0
        elif self.cur_side == 2:
            if self.count < self.path_length:
                self.position[0] -= 1 # moving left
                self.count += 1
            else:
                self.cur_side = (self.cur_side + 1) % 4
                self.count = 0
        else:
            if self.count < self.path_length:
                self.position[1] -= 1 # moving up
                self.count += 1
            else:
                self.cur_side = (self.cur_side + 1) % 4
                self.count = 0