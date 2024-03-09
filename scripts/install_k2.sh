cd extras
git clone https://github.com/k2-fsa/k2.git
cd k2
export K2_MAKE_ARGS="-j6"
python3 setup.py install -f 