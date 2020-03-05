namespace faiss {

    struct SubstructureComputer16 {
        uint64_t a0, a1;

        SubstructureComputer16 () {}

        SubstructureComputer16 (const uint8_t *a8, int code_size) {
            set (a8, code_size);
        }

        void set (const uint8_t *a8, int code_size) {
            assert (code_size == 16);
            const uint64_t *a = (uint64_t *)a8;
            a0 = a[0]; a1 = a[1];
        }

        inline float compute (const uint8_t *b8) const {
            const uint64_t *b = (uint64_t *)b8;
            int accu_num = 0;
            int accu_den = 0;
            accu_num += popcount64 (b[0] & a0) + popcount64 (b[1] & a1);
            accu_den += popcount64 (b[0]) + popcount64 (b[1]);
            if (accu_num == 0)
                return 1.0;
            return 1.0 - (float)(accu_num) / (float)(accu_den);
        }

    };

    struct SubstructureComputer32 {
        uint64_t a0, a1, a2, a3;

        SubstructureComputer32 () {}

        SubstructureComputer32 (const uint8_t *a8, int code_size) {
            set (a8, code_size);
        }

        void set (const uint8_t *a8, int code_size) {
            assert (code_size == 32);
            const uint64_t *a = (uint64_t *)a8;
            a0 = a[0]; a1 = a[1]; a2 = a[2]; a3 = a[3];
        }

        inline float compute (const uint8_t *b8) const {
            const uint64_t *b = (uint64_t *)b8;
            int accu_num = 0;
            int accu_den = 0;
            accu_num += popcount64 (b[0] & a0) + popcount64 (b[1] & a1) +
                        popcount64 (b[2] & a2) + popcount64 (b[3] & a3);
            accu_den += popcount64 (b[0]) + popcount64 (b[1]) +
                        popcount64 (b[2]) + popcount64 (b[3]);
            if (accu_num == 0)
                return 1.0;
            return 1.0 - (float)(accu_num) / (float)(accu_den);
        }

    };

    struct SubstructureComputer64 {
        uint64_t a0, a1, a2, a3, a4, a5, a6, a7;

        SubstructureComputer64 () {}

        SubstructureComputer64 (const uint8_t *a8, int code_size) {
            set (a8, code_size);
        }

        void set (const uint8_t *a8, int code_size) {
            assert (code_size == 64);
            const uint64_t *a = (uint64_t *)a8;
            a0 = a[0]; a1 = a[1]; a2 = a[2]; a3 = a[3];
            a4 = a[4]; a5 = a[5]; a6 = a[6]; a7 = a[7];
        }

        inline float compute (const uint8_t *b8) const {
            const uint64_t *b = (uint64_t *)b8;
            int accu_num = 0;
            int accu_den = 0;
            accu_num += popcount64 (b[0] & a0) + popcount64 (b[1] & a1) +
                        popcount64 (b[2] & a2) + popcount64 (b[3] & a3) +
                        popcount64 (b[4] & a4) + popcount64 (b[5] & a5) +
                        popcount64 (b[6] & a6) + popcount64 (b[7] & a7);
            accu_den += popcount64 (b[0]) + popcount64 (b[1]) +
                        popcount64 (b[2]) + popcount64 (b[3]) +
                        popcount64 (b[4]) + popcount64 (b[5]) +
                        popcount64 (b[6]) + popcount64 (b[7]);
            if (accu_num == 0)
                return 1.0;
            return 1.0 - (float)(accu_num) / (float)(accu_den);
        }

    };

    struct SubstructureComputer128 {
        uint64_t a0, a1, a2, a3, a4, a5, a6, a7,
                a8, a9, a10, a11, a12, a13, a14, a15;

        SubstructureComputer128 () {}

        SubstructureComputer128 (const uint8_t *a8, int code_size) {
            set (a8, code_size);
        }

        void set (const uint8_t *a16, int code_size) {
            assert (code_size == 128 );
            const uint64_t *a = (uint64_t *)a16;
            a0 = a[0]; a1 = a[1]; a2 = a[2]; a3 = a[3];
            a4 = a[4]; a5 = a[5]; a6 = a[6]; a7 = a[7];
            a8 = a[8]; a9 = a[9]; a10 = a[10]; a11 = a[11];
            a12 = a[12]; a13 = a[13]; a14 = a[14]; a15 = a[15];
        }

        inline float compute (const uint8_t *b16) const {
            const uint64_t *b = (uint64_t *)b16;
            int accu_num = 0;
            int accu_den = 0;
            accu_num += popcount64 (b[0] & a0) + popcount64 (b[1] & a1) +
                        popcount64 (b[2] & a2) + popcount64 (b[3] & a3) +
                        popcount64 (b[4] & a4) + popcount64 (b[5] & a5) +
                        popcount64 (b[6] & a6) + popcount64 (b[7] & a7) +
                        popcount64 (b[8] & a8) + popcount64 (b[9] & a9) +
                        popcount64 (b[10] & a10) + popcount64 (b[11] & a11) +
                        popcount64 (b[12] & a12) + popcount64 (b[13] & a13) +
                        popcount64 (b[14] & a14) + popcount64 (b[15] & a15);
            accu_den += popcount64 (b[0]) + popcount64 (b[1]) +
                        popcount64 (b[2]) + popcount64 (b[3]) +
                        popcount64 (b[4]) + popcount64 (b[5]) +
                        popcount64 (b[6]) + popcount64 (b[7]) +
                        popcount64 (b[8]) + popcount64 (b[9]) +
                        popcount64 (b[10]) + popcount64 (b[11]) +
                        popcount64 (b[12]) + popcount64 (b[13]) +
                        popcount64 (b[14]) + popcount64 (b[15]);
            if (accu_num == 0)
                return 1.0;
            return 1.0 - (float)(accu_num) / (float)(accu_den);
        }

    };

    struct SubstructureComputerDefault {
        const uint8_t *a;
        int n;

        SubstructureComputerDefault () {}

        SubstructureComputerDefault (const uint8_t *a8, int code_size) {
            set (a8, code_size);
        }

        void set (const uint8_t *a8, int code_size) {
            a =  a8;
            n = code_size;
        }

        float compute (const uint8_t *b8) const {
            int accu_num = 0;
            int accu_den = 0;
            for (int i = 0; i < n; i++) {
                accu_num += popcount64(a[i] & b8[i]);
                accu_den += popcount64(b8[i]);
            }
            if (accu_num == 0)
                return 1.0;
            return 1.0 - (float)(accu_num) / (float)(accu_den);
        }

    };

// default template
    template<int CODE_SIZE>
    struct SubstructureComputer: SubstructureComputerDefault {
        SubstructureComputer (const uint8_t *a, int code_size):
                SubstructureComputerDefault(a, code_size) {}
    };

#define SPECIALIZED_HC(CODE_SIZE)                     \
    template<> struct SubstructureComputer<CODE_SIZE>:     \
            SubstructureComputer ## CODE_SIZE {            \
        SubstructureComputer (const uint8_t *a):           \
        SubstructureComputer ## CODE_SIZE(a, CODE_SIZE) {} \
    }

    SPECIALIZED_HC(16);
    SPECIALIZED_HC(32);
    SPECIALIZED_HC(64);
    SPECIALIZED_HC(128);

#undef SPECIALIZED_HC

}
