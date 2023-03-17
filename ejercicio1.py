from manim import *
class FourierScene(Scene):
    CONFIG={
        'n_vectors': 2,
        'big_radius': 2,
        'colors': [
            BLUE_D,
            BLUE_A,
            BLUE_C,
            BLUE_B    
        ],
        'tex': r'\rm F',
        'center': ORIGIN,
        'tex_config': {
            'fill_opacity': 0,
            'stroke_width': 1,
            'stroke_color': WHITE,
        },
        'vector_config': {
            'buff': 0,
            'max_tip_length_to_length_ratio': 0.25,
            'tip_length': 0.15,
            'max_stroke_width_to_length_ratio': 10,
            'stroke_width': 1.7,
        },
        'slow_factor': 0.5,
    }
    def construct(self):
        self.slow_factor_tracker=ValueTracker(self.CONFIG['slow_factor'])
        self.vector_clock=ValueTracker(0)
        self.add_vectors()
    def add_vectors(self):
        path=self.get_path()
        coefs=self.get_coefficients_of_path(path)
        vectors=self.get_rotating_vectors(coefficients=coefs)
        self.add(vectors,path)
        self.wait()
    def get_path(self):
        text_mob=Tex(self.CONFIG['tex'], **self.CONFIG['tex_config'])
        text_mob.set(height=2)
        path=text_mob.family_members_with_points()[0]
        return path
    def get_coefficients_of_path(self,path, n_samples=10000, freqs=None):
        if freqs is None:
            freqs=self.get_freqs()
        dt=1/n_samples
        ts=np.arange(0,1,dt)
        samples=np.array([
            path.point_from_proportion(t) for t in ts
        ])
        samples-=self.CONFIG['center']
        complex_samples=samples[:,0]+1j*samples[:,1]
        return [
            np.array([
                np.exp(-TAU*1j*freq*t)*cs
                for t,cs in zip(ts,complex_samples)
            ]).sum()*dt for freq in freqs
        ]
    def get_freqs(self):
        n=self.CONFIG['n_vectors']
        all_freqs=list(range(n//2,-n//2,-1))
        all_freqs.sort(key=abs)
        return all_freqs
    def get_coefficients(self):
        return [complex(0) for _ in range(self.CONFIG['n_vectors'])]
    def get_rotating_vectors(self, freqs=None, coefficients=None):
        vectors=VGroup()
        self.center_tracker=VectorizedPoint(self.CONFIG['center'])
        if freqs is None:
            freqs=self.get_freqs()
        if coefficients is None:
            coefficients=self.get_coefficients()
        last_vector=None
        for freq,coefficient in zip(freqs,coefficients):
            if last_vector is not None:
                center_func= last_vector.get_end()
            else:
                center_func= self.center_tracker.get_location()
            vector=self.get_rotating_vector(
                coefficient=coefficient,
                freq=freq,
                center_func=center_func,
            )
            vectors.add(vector)
            last_vector=vector
        return vectors
    def get_rotating_vector(self,coefficient, freq, center_func):
        vector=Vector(RIGHT, **self.CONFIG['vector_config'])
        vector.scale(abs(coefficient))
        if abs(coefficient)==0:
            phase=0
        else:
            phase=np.log(coefficient).imag
        vector.rotate(phase, about_point=ORIGIN)
        vector.freq=freq
        vector.coefficient=coefficient
        vector.center_func=center_func
        vector.add_updater(self.update_vector)
        return vector
    def update_vector(self,vector, dt):
        time=self.vector_clock.get_value()
        coef=vector.coefficient
        freq=vector.freq
        center_func=vector.center_func
        phase=np.log(coef).imag
        vector.set_length(abs(coef))
        vector.set_angle(TAU*freq*time+phase)
        vector.shift(center_func-vector.get_start())
        return vector