import cv2

def captch_ex(file_name):
  img = cv2.imread(file_name)

  #img_final = cv2.imread(file_name)
  img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  blur = cv2.GaussianBlur(img2gray, (5, 5), 0)
  ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  cv2.imwrite('out_image_1.png', th3)  
  #ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
  
  #image_final = cv2.bitwise_and(img2gray, th3)
  #cv2.imwrite('out_image_2.png', image_final)
  
  #ret, new_img = cv2.threshold(
  #    th3, 180, 255,
  #    cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV
  #cv2.imwrite('out_image_3.png', new_img)
  '''
            line  8 to 12  : Remove noisy portion 
    '''
  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 1))

  # dilate , more the iteration more the dilation
  dilated = cv2.dilate(th3, kernel, iterations=9)

  # get contours
  image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  cv2.imwrite('out_image_4.png', image)
  """
    image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  # cv3.x.x 
    """
  print(len(contours))
  for contour in contours:
    # get rectangle bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)

    # Don't plot small false positives that aren't text
    if w < 35 and h < 35:
      continue

    # draw rectangle around contour on original image
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
    '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg' 
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
  # write original image with added contours to disk
  return img


file_name = 'image.png'
img = captch_ex(file_name)
cv2.imwrite('out_image.png', img)
