
def name_is_img_file(filename: str):
    img_postfix = {'jpg', 'JPG', 'png', 'PNG', 'bmp', 'BMP', 'ppm'}
    postfix = filename[filename.rfind('.')+1:]
    if postfix in img_postfix:
        return True
    else:
        return False


def main():
    filenames = ['1.jpg', '/123/cup.JPG', './qa/qa/cup.png', 'sda/cup.PNG', 'cc.cc.cc.bmp', 'asdwa.sss']
    answers = [True, True, True, True, True, False]
    for filename, answer in zip(filenames, answers):
        assert name_is_img_file(filename) == answer, filename + '   ' + str(answer)
        print(filename, answer)
    print('all passed')


if __name__ == '__main__':
    main()

