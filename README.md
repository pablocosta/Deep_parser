# Deep_parser
An implementation of http://cs.stanford.edu/~danqi/papers/emnlp2014.pdf in python with Keras/Theano library.

--------------------------

Comando para rodar:

python3.4 rede.py

versao do keras = 0.2

O código das features foi inspirado na implementação de https://github.com/jiangfeng1124/acl15-clnndep
e é este:


	for (int i = 2; i >= 0; --i)
	    {
		int index = c.get_stack(i);
		f_word.push_back(get_word_id(c.get_word(index)));
		f_pos.push_back(get_pos_id(c.get_pos(index)));
	    }

	    for (int i = 0; i <= 2; ++i)
	    {
		int index = c.get_buffer(i);
		f_word.push_back(get_word_id(c.get_word(index)));
		f_pos.push_back(get_pos_id(c.get_pos(index)));
	   
	    }

	    for (int i = 0; i <= 1; ++i)
	    {
		int k = c.get_stack(i);

		int index = c.get_left_child(k);
		f_word.push_back(get_word_id(c.get_word(index)));
		f_pos.push_back(get_pos_id(c.get_pos(index)));
		f_label.push_back(get_label_id(c.get_label(index)));
	   
		index = c.get_right_child(k);
		f_word.push_back(get_word_id(c.get_word(index)));
		f_pos.push_back(get_pos_id(c.get_pos(index)));
		f_label.push_back(get_label_id(c.get_label(index)));
	   
		index = c.get_left_child(k, 2);
		f_word.push_back(get_word_id(c.get_word(index)));
		f_pos.push_back(get_pos_id(c.get_pos(index)));
		f_label.push_back(get_label_id(c.get_label(index)));
	   
		index = c.get_right_child(k, 2);
		f_word.push_back(get_word_id(c.get_word(index)));
		f_pos.push_back(get_pos_id(c.get_pos(index)));
		f_label.push_back(get_label_id(c.get_label(index)));
	   
		index = c.get_left_child(c.get_left_child(k));
		f_word.push_back(get_word_id(c.get_word(index)));
		f_pos.push_back(get_pos_id(c.get_pos(index)));
		f_label.push_back(get_label_id(c.get_label(index)));
	   
		index = c.get_right_child(c.get_right_child(k));
		f_word.push_back(get_word_id(c.get_word(index)));
		f_pos.push_back(get_pos_id(c.get_pos(index)));
		f_label.push_back(get_label_id(c.get_label(index)));
	    }
Agora vou por as sub-funcoes:





string Configuration::get_word(int k)
{
    if (k == 0)
        return Config::ROOT;
    else
        -- k;

    return (k < 0 || k >= sent.n)
                ? Config::NIL
                : sent.words[k];
}

/**
 * k starts from 0 (root)
 */

string Configuration::get_pos(int k)
{
    if (k == 0)
        return Config::ROOT;
    else
        -- k;

    return (k < 0 || k >= sent.n)
                ? Config::NIL
                : sent.poss[k];
}



const string & Configuration::get_label(int k)
{
    return tree.get_label(k);
}

const string & DependencyTree::get_label(int k)
{
    if (k <= 0 || k > n)
        return Config::NIL;
    return labels[k];
}


int Configuration::get_left_child(int k, int cnt)
{
    if (k < 0 || k > tree.n)
        return Config::NONEXIST;

    int c = 0;
    for (int i = 1; i < k; ++i)
        if (tree.get_head(i) == k)
            if ((++c) == cnt)
                return i;
    return Config::NONEXIST;
}

int Configuration::get_left_child(int k)
{
    return get_left_child(k, 1);
}

int Configuration::get_right_child(int k, int cnt)
{
    if (k < 0 || k > tree.n)
        return Config::NONEXIST;

    int c = 0;
    for (int i = tree.n; i > k; --i)
        if (tree.get_head(i) == k)
            if ((++c) == cnt)
                return i;
    return Config::NONEXIST;
}

int Configuration::get_right_child(int k)
{
    return get_right_child(k, 1);
}






