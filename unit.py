import unittest

from main import main

class ArgParseTest(unittest.TestCase):

    def test_warmstart(self):
        main("-a NREINFORCE -team output/ntest/end-3x3-team-policy.pt -adv output/ntest/end-3x3-adv-policy.pt -i 1 -ds".split(" "))
        main("-a NREINFORCE -team output/ntest/end-3x3-team-policy.pt -i 1 -ds".split(" "))
        main("-a NREINFORCE -adv output/ntest/end-3x3-adv-policy.pt -i 1".split(" "))
        
        main("-a QREINFORCE -team output/qtest/end-3x3-team-policy.pt -adv output/qtest/end-3x3-adv-policy.pt -i 1 -ds".split(" "))
        main("-a QREINFORCE -team output/qtest/end-3x3-team-policy.pt -i 1 -ds".split(" "))
        main("-a QREINFORCE -adv output/qtest/end-3x3-adv-policy.pt -i 1 -ds".split(" "))

if __name__ == "__main__":
    unittest.main()