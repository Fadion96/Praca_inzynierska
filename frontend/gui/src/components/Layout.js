import React from 'react';
import classnames from 'classnames';
import {
    Button,
    Container,
    Menu,
} from 'semantic-ui-react'

const AppMenu = (props) => {
    const appMenuClass = classnames('app_menu');
    return(
            <Menu inverted icon borderless={true} className={appMenuClass}>
                <Container>
                    <Menu.Item as='a' header>
                        Aplikacja przetwarzająca obrazy
                    </Menu.Item>
                    <Menu.Item>
                        <label className="custom-file-upload">
                        <input type='file' accept='image/*' onChange={props.imagesHandler}/>
                            <i className="file image outline icon"/> Załaduj obraz
                    </label>
                    </Menu.Item>
                     <Menu.Item>
                         <Button
                            content='Click'
                            onClick={props.handleSend}
                         />
                    </Menu.Item>
                </Container>
            </Menu>
    );
};

export default AppMenu
